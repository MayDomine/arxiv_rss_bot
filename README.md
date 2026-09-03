# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-09-03 10:10:36 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [AceSpec: An Asymmetric Edge-Cloud Collaborative Framework for Communication-Efficient LLM Inference](https://arxiv.org/abs/2609.02514)

**Authors**: Yida Zhang, Zhiyong Gao, Shuaibing Yue, Jie Li, Rui Wang  
**Category**: cs.DC  
**Published**: 2026-09-03  
**Score**: 13.5  
**Type**: new  
**ArXiv ID**: 2609.02514v1  

#### Abstract
Deploying Large Language Models (LLMs) on edge devices typically relies on model compression or split inference. However, compression degrades reasoning capabilities, while split inference suffers from severe Wide Area Network (WAN) communication bottlenecks. Edge-cloud speculative decoding emerges ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《AceSpec: An Asymmetric Edge-Cloud Collaborative Framework for Communication-Efficient LLM Inference》总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在边缘设备上部署 **Large Language Models (LLMs)** 面临两大挑战：
- **模型压缩**会损害 LLM 的推理能力；
- **Split Inference** 虽能分担计算，但需频繁传输中间隐藏状态，在高延迟、低带宽的 **Wide Area Network (WAN)** 下形成“通信墙”（Communication Wall）。

尽管 **Speculative Decoding** 通过“草稿-验证”机制减少了通信量，但在 WAN 环境中仍存在严重瓶颈：
- 同步依赖导致“停等”（stop-and-wait）模式；
- 预测被拒绝时引发 **pipeline stall** 和 **网络级回滚（rollback）**，造成性能急剧下降；
- 支持 **nucleus sampling** 会导致上行链路带宽饱和。

---

### 🚀 提出的新方法：AceSpec
作者提出 **AceSpec** —— 一种无需训练、非对称的边云协同框架，核心思想是：
> **用边缘端富余的并行算力换取高昂的 WAN 回滚代价**。

#### 主要创新点：
1. **异步多分支缓存（Asymmetric Multi-Branch Caching）**
   - 利用云验证期间的空闲时间，边缘端主动构建一个 **概率性多分支 token tree cache**；
   - 当预测被拒绝时，系统不再重新生成（redraft），而是通过 **O(1) 局部内存查找** 恢复上下文。

2. **非均匀几何衰减分支分配策略（Non-uniform Geometric Decay Allocation）**
   - 基于 **拉格朗日优化** 的资源分配算法；
   - 在浅层（早期位置）分配更多分支，因为这些位置更易发生拒绝且影响更大；
   - 数学建模为：`F_k ∝ η^k`，其中 `η ∈ (0,1)` 是拓扑衰减因子。

3. **非对称通信协议（Asymmetric Communication Protocol）**
   - 上行仅发送主链 token 的索引（scalar array），极小化上行开销；
   - 下行仅返回稀疏的目标分布（sparsified target distribution）用于局部重采样。

4. **两级约束优化框架**
   - **Network-Aware Budgeting**：动态调整最大树大小 `B*(t)`，确保多分支构造不超出 `RTT + T_verify`；
   - **Task-Aware Shape Allocation**：在预算内最优分配分支数量以最大化缓存命中率。

---

### 🔍 相比现有方法的优势
| 方法 | 缺陷 | AceSpec 如何改进 |
|------|------|----------------|
| Vanilla Speculative Decoding | 同步“停等”，WAN 下效率极低 | 异步流水线 + 缓存避免回滚 |
| PicoSpec / PipeSD / E2-SCI | 单链结构，拒绝即全盘回滚；不支持 nucleus sampling | 多分支缓存支持随机采样，O(1) 恢复 |
| DSSD / Split Inference | 仍依赖密集通信或模型微调 | 无训练需求，通信极轻量化 |
| Tree-based speculation (e.g., SpecInfer) | 通信开销大，不适合 WAN | 仅传主链索引，下行稀疏反馈 |

> ✅ **优势总结**：  
> - 支持 **stochastic nucleus sampling** 而不增加上行流量；  
> - 彻底消除 **network-wide rollback penalty**；  
> - 完全 **training-free**，适用于任意现成 LLM；  
> - 实现 **bandwidth immunity**，即使在极端低带宽下也保持高性能。

---

## 2. 核心实验方法和设置

### 📚 数据集
使用三个具有不同生成复杂度的任务数据集：
- **GSM8K**：多步数学推理任务；
- **HumanEval**：代码生成任务（强调语法精确性）；
- **Alpaca**：通用指令遵循任务。

---

### 💻 实验平台
- **边缘端**：NVIDIA Jetson AGX Orin (32GB)，模拟资源受限边缘节点；
- **云端**：4× NVIDIA A100 (40GB) GPU，运行目标 LLM；
- **网络模拟**：使用 Linux `tc` 工具注入延迟与带宽限制，基准带宽设为 100 Mbps，并测试低至 10 Kbps 的场景。

---

### ⚙️ 模型组合
采用以下 LLM 对进行边云协作：
- Qwen-0.6B (edge) / Qwen-32B (cloud)
- Qwen-1.7B / Qwen-32B
- LLaMA-1B / LLaMA-70B

---

### 📊 评估指标
- **Throughput Speedup**：相对于 autoregressive 基线的吞吐加速比（tokens/s）；
- **End-to-End Latency**：包括 TTFT（Time to First Token）和 ITL（Inter-Token Latency）；
- **Cache Hit Rate**：多分支缓存命中比例；
- **Bandwidth Immunity**：在不同带宽下的性能稳定性；
- **Rollback Penalty**：因拒绝导致的额外延迟。

---

### 🆚 基线方法对比
1. **Autoregressive**：纯云端自回归解码，作为基准；
2. **Vanilla Speculative Decoding**：同步“停等”模式；
3. **Split Inference**：前几层在边缘执行，其余在云端；
4. **DSSD**：分布式分裂推测解码，部分验证下放边缘；
5. **PicoSpec**：单链异步流水线框架。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table I）
| 方法 | Qwen 0.6B/32B (GSM8K) | LLaMA 1B/70B (HumanEval) | 平均表现 |
|------|------------------------|----------------------------|----------|
| Autoregressive | 1.00× | 1.00× | 基准 |
| Vanilla Spec. | 0.57× | 1.45× | 受限严重 |
| Split Inf. | 0.54× | 0.66× | 通信墙明显 |
| PicoSpec | 1.45× | 2.90× | 优于传统 |
| DSSD | 0.85× | 1.95× | 存在瓶颈 |
| **AceSpec (Ours)** | **1.75×** | **3.52×** | ✅ **最高加速** |

> 🔥 **最高达 3.52× 吞吐加速**

---

### 📉 消融实验结果（Table II）
比较三种分支结构：
| 分支策略 | Dataset | TTFT (ms) ↓ | Hit Rate ↑ | Throughput (tok/s) ↑ |
|--------|---------|-------------|------------|-----------------------|
| Single Branch | GSM8K | 442.48 | 0.50 | 20.14 |
| Uniform Branch | GSM8K | 450.79 | 0.94 | 21.58 |
| **Non-Uniform Branch (Ours)** | GSM8K | **436.62** | **0.97** | **24.37** |

> ✅ **非均匀分配显著提升命中率与吞吐，同时降低首 token 延迟**

---

### 🌐 带宽鲁棒性测试（Fig. 5）
- 在 **50 Kbps** 极限带宽下，AceSpec 仍维持接近峰值性能；
- 性能仅在低于 **25 Kbps** 时开始下降；
- 验证了其 **bandwidth immunity** 特性。

---

### 🔬 敏感性分析
- **分支预算 B 增加 → 命中率上升，但边际收益递减**（Fig. 6）；
- **温度 T 提高 → 分布更平 → 接受率下降，但命中率仍高于 81%**（Fig. 7）；
  - 表明 AceSpec 在创造性生成任务中依然有效。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **多分支缓存可将灾难性的网络回滚转化为 O(1) 内存操作**，从根本上解决 WAN 下 speculative decoding 的 pipeline stall 问题；
2. **非均匀分支分配优于均匀策略**，符合 LLM 输出的长尾分布特性；
3. **异步非对称通信协议极大节省带宽**，使系统在低至 50 Kbps 下仍高效运行；
4. **边缘算力可用于“购买”网络容错性**，实现真正的 latency masking；
5. **无需任何模型微调或结构调整**，具备强通用性和部署灵活性。

---

### ⚠️ 局限性
1. **当边缘计算延迟超过 RTT 时，性能可能退化**：
   - 如 Qwen-1.7B 在 HumanEval 上仅得 0.98× 加速；
   - 原因：大 draft model 导致 `T_tree_draft > T_cloud`，成为新的计算瓶颈；
2. **当前实现存在一定的计算开销**（如 tree attention mask 构造）；
3. **假设各 step 之间条件独立**，虽常见于 speculative decoding 分析，但略有简化。

---

### 🔮 未来工作方向
1. **优化 fused attention kernels 和内存管理**，进一步降低多分支构造开销；
2. **引入动态分支剪枝机制**，根据实时接受率调整缓存深度；
3. **扩展到多用户共享服务场景**（multi-tenant serving）；
4. **结合 quantization 或 LoRA 微调进一步提升 draft model 质量**；
5. **探索在移动设备（手机、AR/VR）上的实际部署路径**。

---

## ✅ 总结
**AceSpec** 是首个将 **边云 speculative decoding** 成功适配于 **高延迟、低带宽 WAN 环境** 的通用框架。它通过 **边缘算力换网络鲁棒性**，实现了：
- 最高 **3.52× 吞吐加速**；
- 支持 **nucleus sampling** 的随机生成；
- **50 Kbps 下仍保持近峰性能**；
- **完全无需训练或模型修改**。

该工作为 **LLM 边缘智能** 提供了一条高效、稳健、可扩展的技术路径，推动 LLM 向真实世界边缘应用落地迈进一大步。

</details>

---

### 2. [IDEEA: training-free Input-Dependent stEEring via Activation cluster matching](https://arxiv.org/abs/2609.02089)

**Authors**: Zheng Wang, Muchen Li, Renjie Liao, Yan Leng  
**Category**: cs.CL  
**Published**: 2026-09-03  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.02089v1  

#### Abstract
Steering aligns large language models (LLMs) by injecting a bias into selected activations at inference time, offering a far cheaper alternative to weight-update methods such as supervised fine-tuning or reinforcement learning. However, most existing training-free steering methods are input-independ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：IDEEA: training-free Input-Dependent stEEring via Activation cluster matching**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现有的 **training-free steering** 方法（如 ITI、CAA）通常采用 **input-independent** 的策略，即对所有输入使用同一个固定的 steering direction。然而，不同输入在激活空间中占据不同的区域，其最优的 steering 方向也应不同。单一静态方向无法有效覆盖目标概念的多模态分布，导致部分输入被过度或不足修正，甚至引发“拒绝陷阱”（refusal collapse），即模型倾向于生成“I don’t know”等无信息但“正确”的回答。

### **提出的新方法**
本文提出了 **IDEEA**（Input-Dependent stEEring via Activation cluster matching），一种无需训练的 **input-dependent steering** 框架，其核心思想是：
1. 对每个 attention head 的正负样本激活进行 **聚类**（clustering）；
2. 通过求解一个 **最优匹配问题**（optimal matching），为每一对正负簇构建一组条件方向（cluster-conditional directions）；
3. 在推理时，根据当前输入的激活状态，选择与其最匹配的方向进行 steering。

### **相比现有方法的优势**
- ✅ **输入依赖性**：首次实现真正意义上的 input-dependent steering，避免“一刀切”的静态方向。
- ✅ **保留原始语义**：通过选择与输入激活最对齐的方向，最小化对原始表示的破坏。
- ✅ **避免拒绝陷阱**：显著降低模型因 steering 而陷入“拒绝模式”的风险。
- ✅ **通用性强**：适用于多种 steering 任务（truthfulness, social behavior, political polarity, toxicity mitigation）。
- ✅ **无需额外训练**：完全基于 contrastive activation 数据构建，计算成本远低于 fine-tuning 或 RLHF。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 任务 | 数据集 | 描述 |
|------|--------|------|
| **Truthfulness** | **TruthfulQA** (Lin et al., 2022) | 包含易诱导人类误解的问题，用于测试模型是否生成事实性错误 |
| **Social Behavior** | **Dictator Game** (Leng & Yuan, 2024) | 测试模型在自利、竞争、平等厌恶、社会福利四种行为倾向上的表现 |
| **Political Polarity** | **TwinViews** (Fulay et al., 2024) | 成对的左右倾观点，测试模型能否从左倾转向右倾输出 |
| **Toxicity Mitigation** | **PKU-SafeRLHF + TET** (Ji et al., 2025; Luong et al., 2024) | 使用安全/不安全响应对构建 steering 方向，在 TET 上测试 out-of-distribution 表现 |

### **实验设置和评估指标**
- **模型**：在六种开源指令微调模型上测试：`Llama2 7B`, `Llama3 8B`, `Mistral 7B`, `Qwen2.5 7B`, `Gemma2 2B/9B`。
- **评估协议**：
  - 使用 **LLM judge**（fine-tuned Llama2 7B）对生成答案进行 truthfulness 和 informativeness 打分（0–1）。
  - 主要指标：**truth × info rate (TxI)**，防止模型通过“拒绝回答”来刷高 truth rate。
- **5-fold cross-validation**：超参数在开发集上优化，最终结果取平均值 ± 标准差。

### **基线方法对比**
| 方法 | 类型 | 简介 |
|------|------|------|
| **ITI** (Li et al., 2023) | Input-independent | 使用 top-K attention heads 的 mass-mean direction |
| **CAA** (Rimsky et al., 2024) | Input-independent | 在整层 residual stream 添加 mass-mean direction |
| **SAE** (Bricken et al., 2023) | Feature-based | 使用预训练 SAE 提取的单义特征进行 steering |
| **SEA** (Qiu et al., 2024) | Subspace-based | 通过子空间投影最大化正样本协方差，最小化负样本协方差 |

本文提出两种 IDEEA 变体：
- **min-perp**: 选择与当前激活最对齐的方向（最小垂直距离）
- **nearest-cluster**: 将当前激活分配到最近的负簇，并使用对应匹配方向

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（TruthfulQA）**
在 **TruthfulQA** 上，IDEEA 显著优于所有基线：

| 方法 | 平均 TxI 提升（vs. base） |
|------|--------------------------|
| **base** | 0% |
| **ITI** | ~9.9% |
| **SEA** | 17.4% |
| **IDEEA (min-perp)** | **+34.2%** |
| **IDEEA (nearest-cluster)** | **+28.5%** |

> 🔺 **最高提升达 23.5%**（单个模型上），平均提升 **9.9%** 超过最佳 input-independent 基线。

### **与其他任务的结果**
| 任务 | 方法 | 结果 |
|------|------|------|
| **Dictator Game** | IDEEA (nearest-cluster) | 社会行为信号增益 **+292.1%**，超过 sys prompt weak（未泄露决策规则） |
| **Political Polarity** | IDEEA (min-perp) | 成功率 **50.6%**，远超 ITI (36.6%) 和 CAA (45.6%) |
| **Toxicity Mitigation** | IDEEA (nearest-cluster) | 安全率 **84.7%**，仅次于 SAE (91.7%)，但 SAE 泛化性差 |

### **消融实验结果**
#### **(1) 聚类数 $n_c$ 的影响**
- 性能随 $n_c$ 增加先上升后饱和，**最优在 $n_c=5-6$**。
- 即使 $n_c=2$，IDEEA 仍优于 ITI，说明聚类本身即可带来增益。

#### **(2) 匹配机制的重要性（ablation: nearest-pos-neg）**
- 若跳过最优匹配，允许任意正负簇组合（nearest-pos-neg），性能下降 **平均 3.3%**，最大下降 **6.6%**。
- 说明 **QAP 最优匹配提供了重要正则化**，防止选择次优方向。

#### **(3) 固定簇数对称性（ablation: auto-nc）**
- 允许正负簇数不同（由 Silhouette Score 自动选择），性能进一步下降 **平均 4.1%**。
- 说明 **结构约束（对称簇数 + 双射匹配）有助于稳定方向空间**。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **activation space 是多模态的**：同一概念（如 truthfulness）在激活空间中分布在多个子区域，单一方向无法覆盖全部。
2. ✅ **input-dependent steering 更有效**：根据输入动态选择 steering 方向，能更精准地引导模型行为。
3. ✅ **避免拒绝陷阱**：IDEEA 显著降低“拒绝回答”比例（如 ITI 拒绝率达 31%，而 IDEEA min-perp 降至 8%）。
4. ✅ **几何视角解释成功原因**：如图3所示，cluster-based steering 能将负样本分别映射到多个正样本簇，而 ITI 只能覆盖一部分。

### **方法的局限性**
- ⚠️ **跨层漂移问题**（inter-layer drift）：早期层的 steering 会改变后续层的激活分布，可能导致后续层的 cluster fit 不再适用。
- ⚠️ **依赖 contrastive 数据质量**：若 contrastive prompts 不够代表性，聚类可能失效。
- ⚠️ **计算复杂度**：虽然 training-free，但 QAP 最优匹配为 NP-hard，虽小规模可解，但难以扩展到极大簇数。

### **未来工作方向**
- 🔮 结合 **CAA 的单层干预** 与 IDEEA 的 input-dependent 选择，缓解跨层漂移。
- 🔮 探索 **online adaptation**：在生成过程中动态更新 steering 方向。
- 🔮 将 IDEEA 思想应用于 **其他 steering 范式**，如 SAE 或 SEA。
- 🔮 研究如何 **自动确定最优 $n_c$**，而非手动搜索。

---

> 💡 **一句话总结**：  
> **IDEEA 通过聚类 + 最优匹配构建输入依赖的 steering 方向，在无需训练的前提下，显著提升了 steering 效果，避免了传统方法的“拒绝陷阱”，并揭示了 activation space 的多模态本质。**

</details>

---

### 3. [Scaling Inference Prefill with High-Radix Photonic Interconnects](https://arxiv.org/abs/2609.01821)

**Authors**: Arulselvan Madhavan, Peter Carson, Taylor Groves, Thomas Graham  
**Category**: cs.DC  
**Published**: 2026-09-03  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.01821v1  

#### Abstract
With the rise of inference as today's dominant AI workload, the industry is transitioning to high-bandwidth photonic interconnects to meet the large scale-up requirements of increasingly complex Mixture-of-Experts (MoE) models. This paper quantifies the benefits of 3D-integrated photonic interconnec...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Scaling Inference Prefill with High-Radix Photonic Interconnects》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
随着 AI 推理成为主导的计算负载（占总计算周期的 80–90%），尤其是 **Large Language Model (LLM)** 的广泛应用，推理中的 **prefill 阶段**（即处理输入 prompt 并生成初始 KV Cache）已成为系统性能的关键瓶颈。特别是在以下场景中：
- **高并发聊天服务**：需要高吞吐量；
- **推理型与代理型 AI 工作流**（reasoning and agentic AI）：需要支持超长上下文（如 128K 到 1M tokens）。

传统基于铜缆（copper-based）的互连技术受限于带宽、距离和功耗，在大规模分布式推理中面临“scale-up pod”无法跨机架扩展的问题，导致通信开销急剧上升。

### 提出了什么新方法或新思路
本文提出并量化了使用 **3D-integrated photonic interconnects**（三维集成光子互连）来优化 LLM 推理 prefill 阶段的方法，其核心是构建一个 **high-radix、high-bandwidth、多机架可扩展的 scale-up pod**。

具体创新点包括：
- 利用 **硅光子学 (Silicon Photonics)** 和 **共封装光学 (Co-Packaged Optics, CPO)** 技术实现高达 **64 Tb/s 双向带宽/GPU**；
- 实现 **radix 扩展至 1152**，远超铜缆限制；
- 构建跨越多个机架的统一高性能互连域（scale-up pod），避免落入低速 scale-out fabric；
- 在建模层面结合 XLA 成本模型与光互连参数，进行端到端 prefill 延迟分析。

### 相比现有方法的优势
| 维度 | 传统铜缆互连 | 光子互连（本文方案） |
|------|----------------|------------------------|
| **Reach** | ~1m @ 224Gbps/lane | 达 **1000m**，支持跨机架 |
| **Port Pitch** | 1020μm（受限于电气信号完整性） | **127μm bidirectional fiber**，密度提升 8× |
| **Bandwidth/fiber** | ~0.224 Tbps/Diff Pair | **1.79 Tbps/fiber**（WDM 多波长复用） |
| **Energy Efficiency** | ~5 pJ/bit（passive copper） | **4.3 pJ/bit**（完整链路），其中 PIC+Laser 仅 2.3 pJ/bit |
| **Scale-up Pod Size** | 单机架内（通常 ≤72–288 GPUs） | 支持 **1152-GPU 跨机架 pod** |

优势总结：
- 显著降低通信延迟，尤其在 batch size 大、context length 长时；
- 支持更大规模并行，满足严格 TTFT（Time-to-First-Token）SLA；
- 更优能效比，适合数据中心部署。

---

## 2. 核心实验方法和设置

### 使用了哪些模型（非数据集）
由于研究聚焦于 **AI 推理架构性能建模**，并未使用传统意义上的“数据集”，而是基于三种典型的 **Mixture-of-Experts (MoE)** 模型配置进行仿真：

| Context Length | 描述 |
|---------------|------|
| Short Context | 1K–8K tokens（典型对话场景） |
| Medium Context | 128K tokens（复杂推理、代码生成） |
| Long Context | 1M tokens（极端代理任务） |

使用的代表性模型为 **DeepSeek R1 variant**（42B active parameters, FP4 quantization）。

### 实验设置和评估指标

#### 硬件平台对比
在四种硬件平台上进行模拟比较：
- **B200**: NVIDIA 当前 GPU，受限于单 rack scale-up；
- **B300**: 下一代平台，假设更高 FLOPs；
- **Rubin**: 假设高度并行化设计（NVIDIA 路线图推测）；
- **R4**: 自研 hypothetical 平台，具备原生电光混合 scale-up pod（576 GPUs）；

#### 并行策略（Parallelism Strategy）
采用 **3D 并行**：
- **x-axis**: Sequence/Context-Parallel（跨机架分割序列维度）；
- **y-axis**: Expert + Tensor Parallel（带宽最密集轴，all-to-all / all-gather）；
- **z-axis**: FFW Tensor-Parallel（前馈网络切分）；

例如：`4×72×1` 表示 4 个机架 × 72 个设备 × 1 层级。

#### 评估方式
- **Device Sweep**：固定总 token 数，变化设备数量（从 8 到 1152），寻找最优 mesh；
- **Batch Sweep**：固定设备数，变化 batch size（最高达 16,384），观察吞吐与延迟关系；
- **关键指标**：
  - **Overlapped Prefill Latency**（重叠后的预填充延迟）
  - **Input-Token Throughput**
  - **TTFT (Time-to-First-Token)**

#### 基线方法对比
- **Electrical Baseline**：当前主流铜缆互连系统（如 NVLink + InfiniBand）；
- **Optical Comparison**：假设启用 4× 带宽、radix 高达 1152 的全光互连系统；
- 对比指标为 **latency ratio = Electrical / Optical**，大于 1 表示光学改进。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table IV 和正文）

| Context Length | Platform | Latency Improvement (×) | 条件说明 |
|----------------|----------|----------------------------|-----------|
| 1K tokens      | B300     | ~2.6×                      | 高 batch，compute-bound 减弱 |
| 8K tokens      | B300     | ~2.2×                      | 同上 |
| 128K tokens    | B300     | ~2.9×                      | 通信压力显著增加 |
| 1M tokens      | B300     | ~2.3×                      | 极端上下文，依赖跨机架通信 |
| 128K tokens    | B200/Rubin | **~4.3–5.8×**             | 因原生 scale-up pod 小，跨 rack 惩罚严重 |
| 1M tokens      | B200/Rubin | **~3.4–4.5×**             | 光互连避免降级至 scale-out fabric |
| 1M tokens      | R4       | **up to 8.5×**              | 原生大 pod + 光加速叠加效应 |

更一般地，论文报告：
- 在 **stressed high-batch regimes** 中：**2.1–3.2× 延迟改善**；
- 在 **communication-limited configurations** 中：**2.8–5.8× 改善**；
- 在 **1152-GPU 规模下**，光学 enable 了原本因电气限制不可达的配置，带来 **2.2–4.5× 加速**。

### 与基线方法的对比结果
- 所有平台均受益，但 **越受 scale-up pod 限制的平台增益越大**（如 B200 > B300 > R4）；
- **FP4 比 FP8 增益更大**：因为更快的算术使通信成为瓶颈，放大光互连价值；
- 在 **medium-to-long context** 场景下，通信占比超过 65%，因此带宽提升直接转化为延迟下降；
- 图表显示：光学互连下，**吞吐随 batch 增加持续上升而延迟不剧增**，体现更强的可扩展性。

### 消融实验结果（隐含分析）
虽然未明确命名“ablation study”，但通过多维 sweep 得出以下因果结论：
- 当 workload **尚未跨越 scale-up boundary**（如 128K @ 16–72 devices）时，增益较小 → 说明只有当通信真正受限时才显现优势；
- **compute-bound 场景增益小**，**communication-bound 场景增益大** → 验证了光互连的价值边界；
- 不同 mesh shape 的 exhaustive search 显示：最大化 y-axis（expert+tensor parallel）对性能最关键。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **Prefill 是长上下文推理的主要瓶颈**，且其性能越来越受制于互连而非算力；
2. ✅ **3D-integrated photonic interconnects 能有效打破 scale-up pod 的物理限制**，将 high-bandwidth 连接延伸至千级 GPU 跨机架系统；
3. ✅ **光学互连带来的延迟改善是非均匀的**：在通信密集型、大批量、长上下文、小原生 pod 的系统中最显著；
4. ✅ **radix 提升至 1152 和带宽提升 4× 可解锁新的系统设计空间**，使得 1M-token 推理达到实用 TTFT；
5. ✅ **能量效率也显著优于传统 pluggable optics**，接近 passive copper 水平（4.3 pJ/bit），具备部署可行性。

### 方法的局限性
- ❗ **纯分析性建模**：基于 XLA 成本模型外推，尚未在真实光子硬件上验证；
- ❗ **忽略实际工程挑战**：未考虑 link latency、thermal effect、signal integrity、deployment cost 和 TCO；
- ❗ **focus on prefill-centric analysis**：decode 阶段可能成为新的瓶颈（如 memory bandwidth limited），未实现 co-design；
- ❗ **假设理想 4× 带宽无损耗**，现实中可能存在误码率、同步等问题。

### 未来工作方向
1. 🔮 **full serving-system co-design**：联合优化 prefill（compute-bound）与 decode（memory-bound）资源配置；
2. 🔮 **探索光互连支持的新功能**：
   - 分布式 KV Cache placement
   - Prefix-cache sharing
   - 快速 prefill-to-decode KV transfer
   - 故障 GPU 的拓扑重构
3. 🔮 **实测验证**：在光学硬件原型上运行真实 workload，校准模型预测；
4. 🔮 **成本与 TCO 分析**：纳入光模块、冷却、维护等综合因素，判断经济可行性。

---

> 📝 **备注**：文中 AI 工具仅用于编辑与格式化辅助，所有技术主张、数据与结论均由作者负责。

</details>

---

### 4. [Scalable Bayesian Optimization of Composite Functions for Image-Based Inverse Problems in Materials Characterization](https://arxiv.org/abs/2609.02126)

**Authors**: Dasol Yoon, Poompol Buathong, Chia-Hao Lee, Yujia Zhang, David A. Muller, Peter I. Frazier  
**Category**: cs.LG  
**Published**: 2026-09-03  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.02126v1  

#### Abstract
Estimating physical parameters from scientific images is a common inverse problem in materials characterization that often relies on expensive physics-based simulations. In electron microscopy, specimen thickness and crystal mistilt are critical parameters that govern how electrons scatter through t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Scalable Bayesian Optimization of Composite Functions for Image-Based Inverse Problems in Materials Characterization

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在材料表征领域，**从科学图像（如电子显微镜中的 PACBED 图像）中估计物理参数**是一个典型的逆问题。这类问题通常依赖昂贵的基于物理的模拟器（如 multislice simulation），且目标函数具有高维输出（如数万个像素）。传统方法面临以下挑战：
- **Grid search**：计算成本随参数空间指数增长，效率低下。
- **Neural network 方法**：需要大量预训练数据，泛化能力差，难以适应新条件。

### 提出的新方法：SBOCF
本文提出 **Scalable Bayesian Optimization of Composite Functions (SBOCF)**，一种面向复合函数结构的可扩展贝叶斯优化框架，用于解决上述高维、昂贵模拟下的逆问题。

#### 核心创新点：
- **利用复合函数结构（Composite Function Structure）**  
  将目标函数 $ f(x) = \text{SSE}(y_{\text{sim}}(x), y_{\text{exp}}) $ 显式建模为两步过程：先通过模拟器生成图像 $ y(x) $，再通过已知函数（如 SSE）计算误差。SBOCF 利用这一结构进行更高效的不确定性传播。
  
- **引入 Patch-Level Summary Representation**  
  将原始 $157\times157$（共 24,649 像素）的 PACBED 图像划分为 $3\times3$ 共 9 个 patch，对每个 patch 内像素求和作为 summary，将中间输出维度从 **24,649 降至 9**。

- **双校正项建模聚合误差**  
  引入两个额外变量：
  - $ \epsilon(x) $：输入相关的乘法校正项
  - $ \delta(x) $：残差项  
  构造新的复合形式：  
  $$
  f(x) = \epsilon(x) \cdot f_p(x) + \delta(x)
  $$
  其中 $ f_p(x) $ 是 patch-level SSE。这使得模型既能保留原始 pixel-wise SSE 目标，又能大幅降低建模复杂度。

- **最终建模维度仅需 11 个 GP 输出**  
  即：9 个 patch summaries + $ \epsilon(x) $ + $ \delta(x) $

### 相比现有方法的优势
| 方法 | 缺点 | SBOCF 改进 |
|------|------|------------|
| Standard BO (EI/KG) | 忽略中间图像结构，仅建模 scalar SSE，采样效率低 | 利用复合结构提升信息利用率 |
| BOCF [11] | 需建模全部像素强度（>24k outputs），计算不可行 | 降维至 11 outputs，实现可扩展性 |
| CNN-based 方法 [6–8] | 需大量预训练数据，缺乏通用性 | 无需训练数据，适用于新样品/条件 |

---

## 2. 核心实验方法和设置

### 数据集
- **Synthetic Benchmarks**（仿真数据）：
  - **Sim380**, **Sim100**, **Sim200**：SrTiO₃ 样品，真实厚度分别为 380Å, 100Å, 200Å，倾斜角为 (±1.5, ±3, ∓5) mrad
  - 使用 **abTEM** 包在线生成 PACBED 图像（beam energy: 200 keV, resolution: 157×157）
- **Experimental Benchmark**：
  - **Exp380**：来自文献 [6] 的真实 SrTiO₃ 实验 PACBED 数据
- **下游任务验证**：
  - 在 Sim200 上进行 **multislice electron ptychography (MEP)** 重建，检验参数估计对成像质量的影响

### 实验设置
- **优化变量**：3 维参数空间
  - Specimen thickness: [5, 500] Å
  - Beam tilt-x 和 tilt-y: [-10, 10] mrad
- **预算限制**：最多 **50 次模拟评估**
- **初始化**：7 点 Sobol 序列
- **重复次数**：20 次独立试验取平均
- **GP 设置**：
  - 均值函数：constant
  - 协方差核：Matérn-5/2
  - 超参数通过 MLE 学习

### 评估指标
- **主指标**：
  - 最佳观测到的 **pixel-wise Sum of Squared Errors (SSE)**
  - 参数估计误差（与 ground truth 或参考值之间的绝对偏差）
- **辅助指标**：
  - 获取函数（acquisition function）每轮运行时间
  - 性能增益比（median baseline / median SBOCF）

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **BO(EI)** | 标准贝叶斯优化 + Expected Improvement |
| **BO(KG)** | 标准贝叶斯优化 + Knowledge Gradient |
| **Random Sampling** | 随机搜索作为下界基准 |
| **SBOCF (Ours)** | 所提方法，使用 $3\times3$ 分块 + 双校正项 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Sim380，无噪声）

| 指标 | SBOCF vs BO(EI) | SBOCF vs BO(KG) |
|------|------------------|------------------|
| **Final SSE 改善倍数** | **290× 更低** | **377× 更低** |
| **Thickness 误差改善** | 24× | 23× |
| **Tilt-X 误差改善** | 26× | 22× |
| **Tilt-Y 误差改善** | 35× | 51× |

> ✅ SBOCF 在所有指标上显著优于基线，尤其在厚样本（Sim380）中优势巨大。

### 其他合成数据集表现

| 数据集 | Final SSE 改善倍数 (vs BO(EI)/BO(KG)) |
|--------|---------------------------------------|
| **Sim100** | 47× / 320× |
| **Sim200** | 37× / 202× |

尽管薄样本（如 Sim100）信噪比较低导致增益缩小，SBOCF 仍保持明显优势。

### 实验数据（Exp380）结果
- **SSE 改善**：
  - vs BO(EI): **1.08×**
  - vs BO(KG): **1.13×**
- **参数估计一致性更高**：
  - 厚度估计接近 380 Å（参考值）
  - 倾斜角估计与文献 [6] 报道一致
- **无需预训练即可达到竞争性性能**

### 下游 MEP 重建效果
- 使用 SBOCF 估计的参数进行 **multislice electron ptychography** 重建：
  - 成功恢复出 **sharp、垂直的原子柱**（图6 d-f）
  - 若忽略 mistilt，则原子柱被拉长并倾斜（图6 a-c）
- 参数精度：厚度误差 < 5 Å，倾斜角误差 < 0.1 mrad

### 消融实验结果

#### （1）噪声鲁棒性测试（Poisson Noise）
- 添加不同水平 Poisson 噪声（peak photon count: 150, 100, 50）
- 发现：
  - 随着噪声增加，SBOCF 相对优势减弱（因像素级结构被破坏）
  - 但在 **厚样本（Sim380）中仍保持稳定优势**
  - 薄样本（Sim100/200）对噪声更敏感，符合 PACBED 物理特性认知

#### （2）分块策略对比（Ablation Study on Partition Pattern）
- 对比两种图像划分方式：
  - **Square Partition**（$3\times3$ 网格）✅
  - **Domain Partition**（中心圆 + 四象限，按强度分区）
- 结果显示：
  - **Square 分块在几乎所有设置下表现更好或相当**
  - Domain 分块可能过于粗粒度，丢失局部空间信息
  - 表明 **patch 设计影响性能，值得进一步研究自适应分块**

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **SBOCF 显著提升了 BO 在图像逆问题中的采样效率**  
   在仅 50 次模拟预算下，将最终 SSE 降低达 **290×**，远超标准 BO。
   
2. ✅ **通过 patch-level summary + correction terms 成功实现了可扩展性**  
   将建模输出从 >24k 降至 **11 个 GP 输出**，使 BOCF 能应用于高分辨率图像。

3. ✅ **无需训练数据即可获得与 CNN 方法相当甚至更优的参数估计**  
   在实验数据上估计结果与已有报道一致，具备良好实用性。

4. ✅ **准确的参数估计显著提升下游成像质量**  
   在 MEP 中纠正 mistilt 后，成功恢复锐利原子结构。

5. 🔍 **厚样本比薄样本更适合 PACBED 参数估计**  
   因其 PACBED 图案包含更多可靠结构信息，抗噪能力强。

### 方法局限性
- 当前方法假设模拟器是可信的（即 simulator-to-reality gap 较小）
- 对极高噪声或严重模型失配情况性能下降
- 分块模式固定，未考虑图像内容自适应调整
- 当前仅处理 3 个参数，扩展至更多参数（如 aberrations）需进一步验证

### 未来工作方向
- 扩展至更多 PACBED 敏感参数：probe aberrations、atomic vibration amplitudes
- 探索 **adaptive patch partitioning** 策略（如基于注意力机制）
- 结合 **multi-fidelity modeling** 加速收敛
- 推广至其他科学成像领域的逆问题（如 X-ray diffraction, cryo-EM）

---

> 📌 **一句话总结**：  
> SBOCF 通过巧妙利用图像的局部结构与复合函数形式，在不牺牲原始目标的前提下，将高维图像逆问题转化为低维代理学习任务，实现了在极有限模拟预算下的高效、精准参数估计，为材料表征等依赖昂贵仿真的科学问题提供了强有力的工具。

</details>

---

### 5. [DMRL: Document-Mediated Reinforcement Learning for Skill Optimization in Advertising Recommendation](https://arxiv.org/abs/2609.02170)

**Authors**: Wei Zhang, Hongji Li, Song Sun, Peng Yu, Xue Yang, Lei Zhao, Peng Jiang  
**Category**: cs.LG  
**Published**: 2026-09-03  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.02170v1  

#### Abstract
Advertising recommendation requires continuously tuning complex system parameters while balancing commercial returns and user experience. Recent work has introduced large language models (LLMs) with skill documents to assist this labor-intensive process, but skill optimization remains largely prompt...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《DMRL: Document-Mediated Reinforcement Learning for Skill Optimization in Advertising Recommendation》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在线广告推荐系统中，参数调优通常依赖人工经验或超参数优化（HPO），存在以下三大瓶颈：
- **人力成本高**：调参效率低，难以规模化；
- **知识难以沉淀**：专家经验难以形式化、复用；
- **探索不系统**：易陷入局部最优。

尽管已有研究引入大语言模型（LLM）通过 **Skill Documents** 辅助调参，但现有方法仍以文本提示（prompt-driven）为主，缺乏对具体文档编辑行为的**奖励归因机制**（credit assignment）。此外，广告系统的最终目标（如LTV）具有显著延迟（delayed feedback），且不同用户群体对短期信号到长期收益的映射关系差异巨大（population heterogeneity），导致传统方法难以准确建模长期回报。

### 提出的新方法与新思路
本文提出 **Document-Mediated Reinforcement Learning (DMRL)**，一种面向技能自演化的强化学习框架，其核心思想是将技能优化过程解耦为两个层级：

- **上层**：Skill Optimizer 负责对结构化技能文档进行可控编辑；
- **下层**：Frozen Task Agent 根据修改后的技能文档执行真实 A/B 测试；
- **媒介**：技能文档作为语义接口连接上下两层。

在此基础上，提出两大关键技术模块：

#### (1) Dual-Relative Policy Optimization (DRPO)
一种后训练策略优化算法，用于更鲁棒地估计编辑动作的优势函数（advantage estimation），具备三个关键设计：
- **MAD-based Normalization**：使用中位数绝对偏差（MAD）替代标准差，提升对异常奖励的鲁棒性；
- **Reference Reward from Control Group**：利用对照组表现作为基准，增强优势估计的可靠性；
- **Edit Cost Regularization**：引入编辑代价惩罚项，鼓励最小化破坏性的有效修改，实现风险感知的技能演化。

#### (2) Long-term Reward Predictor (LRP)
一个预测模块，用于从短期反馈信号中估计长期奖励，解决高延迟问题。其创新在于：
- **Disentangled Representation Learning**：将用户表征解耦为群体不变（population-agnostic）和群体特异（population-specific）两部分；
- **Cross-Attention Historical Transfer**：构建历史实验记忆库，通过 cross-attention 检索相似情境下的早期响应模式，实现跨实验知识迁移。

#### 两阶段训练策略（Two-Stage Training）
- 第一阶段：固定策略网络，预训练 LRP；
- 第二阶段：冻结 LRP，训练 DRPO 策略；
该策略提升了整个系统的稳定性与泛化能力。

### 相比现有方法的优势
| 维度 | 现有方法局限 | DMRL 改进 |
|------|--------------|----------|
| 编辑归因 | 缺乏结构化编辑与奖励之间的因果关联 | 显式建模编辑动作 → 回报路径，支持 traceable 修改 |
| 长期回报建模 | 忽视群体异质性，统一建模所有用户 | 显式建模 population-invariant 与 population-specific 动态 |
| 延迟反馈处理 | 多基于即时信号或简单代理变量 | 利用历史转移机制，精准预测延迟 LTV/PES |
| 安全性与可控性 | 易产生激进、不稳定更新 | 引入 edit cost 正则化，促进稳健演化 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- 数据来源：快手（Kuaishou）大规模短视频广告平台的真实线上流量日志；
- 构造方式：低频采样约 **1200万条用户轨迹**，每条包含：
  - 参数干预记录（△a）
  - 用户特征（u）
  - 短期反馈（rs，1天内）
  - 长期结果（rt，7天内）
  - 所属人群标签（population label）

#### 用户分群依据（基于短/长信号表现）：
| 分组 | 描述 |
|------|------|
| Low-ST Low-LT | 短期&长期指标均低 |
| High-ST Low-LT | 短期高但长期低（可能“刷量”） |
| Low-ST High-LT | 短期低但长期高（潜在优质用户） |
| High-ST High-LT | 双高优质用户 |

> 数据按时间顺序划分训练/验证/测试集，防止时序泄露；记忆库存储已完成实验的历史记录。

### 实验设置
- **上层模型**：Qwen3-8B 作为 optimizer model；
- **下层任务代理**：OpenAI Codex（基于 GPT-3.5）作为 frozen Task Agent；
- **编辑生成**：每次迭代生成 8 个候选编辑（rollouts）；
- **部署环境**：真实线上 A/B 实验平台，Task Agent 在安全白名单范围内调整参数。

### 评估指标
| 指标 | 含义 | 时间窗口 |
|------|------|---------|
| **AUD@1d / AUD@7d** | App Usage Duration，衡量用户参与度 | 1天 / 7天 |
| **PES@1d / PES@7d** | Posterior Expected Spend，广告变现能力 | 1天 / 7天 |
| **LTV** | Life Time Value，综合经济价值（含 engagement + monetization） | 7天累计 |

> 所有增益以百分比表示，**LTV > +0.01% 视为实际有意义改进**；采用 CUPED 技术降低方差，确保统计显著性。

### 基线方法对比
分为两类进行公平比较（控制变量法）：

#### (1) Skill Optimization 方法（固定 LRP）
- LRP + SAGE
- LRP + SkillOpt
- LRP + SKILLRL

#### (2) Long-term Reward Modeling 方法（固定 DRPO）
- DRPO + TFT（Temporal Fusion Transformer）
- DRPO + DelayAdapter

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 2）

| 方法 | AUD@1d | PES@1d | AUD@7d | PES@7d | **LTV** |
|------|--------|--------|--------|--------|--------|
| SAGE | +0.049% | +0.757% | -0.094% | -0.911% | **-0.049%** |
| SkillOpt | +0.012% | -0.515% | +0.069% | +0.548% | **+0.024%** |
| SKILLRL | +0.101% | -1.547% | +0.133% | -0.430% | **+0.004%** |
| TFT | -0.187% | -0.085% | +0.018% | +0.656% | **+0.015%** |
| DelayAdapter | -0.158% | +1.480% | -0.060% | +0.134% | **+0.005%** |
| **DMRL (Ours)** | **-0.020%** | **+0.644%** | **+0.082%** | **+0.960%** | **+0.052%** ✅ |

> ✅ DMRL 在 **LTV 上达到 +0.052%**，显著优于所有基线，且在 PES@7d 上取得最大增益。

### 与基线对比结果
- 相比最强 skill optimization 基线 SKILLRL，DMRL 将 LTV 提升 **+0.048pp**；
- 相比最佳 reward modeling 基线 TFT，DMRL 提升 **+0.037pp LTV**；
- DMRL 实现了**更均衡的 trade-off**：既避免了 SAGE 对长期指标的损害，也克服了 TFT 对短期体验的负面影响。

### 消融实验结果

#### (1) LRP 模块消融（Table 3）
| 变体 | PES@7d | LTV |
|------|-------|-----|
| UE（统一编码器） | +0.257% | +0.031% |
| w/o MB（无记忆库） | +0.491% | +0.020% |
| UE w/o MB | -0.231% | **-0.028%** |
| **Full LRP** | **+0.662%** | **+0.042%** |

> 结果表明：**disentangled encoding + memory bank** 是关键，二者协同才能稳定提升长期变现与总体价值。

#### (2) DRPO 模块消融（Table 4）
| 变体 | PES@7d | LTV |
|------|-------|-----|
| w/o MN（无 MAD 归一化） | -1.177% | +0.008% |
| w/o RR（无对照组参考） | -0.371% | +0.013% |
| w/o EC（无编辑代价） | -0.688% | **-0.018%** |
| Vanilla GRPO | -1.216% | **-0.069%** |
| **DRPO (Ours)** | **+0.059%** | **+0.021%** |

> 移除任一组件都会导致长期收益下降，尤其 **edit cost** 对防止过度修改至关重要。

#### (3) 两阶段训练策略（Table 5）
| 策略 | PES@7d | LTV |
|------|-------|-----|
| 单阶段联合训练 | -0.702% | **-0.039%** |
| **两阶段训练（Ours）** | **+0.960%** | **+0.052%** ✅ |

> 联合训练会因 reward signal 不稳定而导致负向优化，**分阶段训练显著提升稳定性与效果**。

#### (4) 模型规模与 rollout 数量敏感性分析
- **Backbone 规模**（Table 6）：Qwen3-8B 表现最佳（LTV +0.064%），过大（14B）或过小（0.6B）均性能下降；
- **Rollout 数量**（Table 7）：8 个 rollouts 效果最好（LTV +0.050%），过多（10）或过少（4）均劣化。

---

## 4. 关键结论和发现

### 主要发现
1. **技能文档可作为可操作的语义接口**，支持结构化编辑与策略优化闭环；
2. **显式的编辑代价建模（edit cost）能有效抑制无效或破坏性修改**，提升策略安全性；
3. **长期回报预测必须考虑 population heterogeneity**，否则会导致少数群体被忽视；
4. **历史动态模式可通过 cross-attention 进行有效迁移**，增强冷启动与稀疏场景下的预测能力；
5. **两阶段训练策略对于复杂延迟反馈系统至关重要**，先稳 reward 再训 policy 更可靠。

### 方法的局限性
- 当前技能文档格式仍需一定程度的手工定义，尚未完全自动化构建；
- 编辑空间受限于预设的结构模板（location + action + content），灵活性有限；
- 对极端罕见人群（如 <1% 流量）的建模仍具挑战；
- 实验周期较长，依赖大量 A/B 测试资源。

### 未来工作方向
- 探索多目标优化框架，平衡用户体验、平台收入与广告主 ROI；
- 引入 **fairness-aware objectives**，主动保护弱势用户群体利益；
- 发展 **privacy-preserving training mechanisms**，如联邦学习或差分隐私；
- 扩展至动态变化的流量分布场景，研究持续适应机制（continual adaptation）；
- 探索自动技能发现与初始化技术，减少人工先验依赖。

---

> 💡 **总结一句话**：  
> DMRL 通过 **结构化解耦 + 鲁棒优势估计 + 群体感知长期预测**，实现了广告推荐系统中技能文档的自主、安全、高效演化，在真实工业场景中取得了显著优于现有方法的综合经济效益。

</details>

---

### 6. [Unfolding the Leech Lattice: Fused Multi-Shell Decoding and VRAM Layouts for 2-Bit LLM Weights](https://arxiv.org/abs/2609.02652)

**Authors**: Pier-Jean Malandrino (Scub)  
**Category**: cs.LG  
**Published**: 2026-09-03  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.02652v1  

#### Abstract
Leech-lattice vector quantization holds the strongest reported 2-bit quality under its own evaluation protocol. Its kernel decodes one shell; we found no implementation of the multi-shell decoder the rate requires. This paper supplies one and measures its serving cost for decode-phase GEMV at batch ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Unfolding the Leech Lattice: Fused Multi-Shell Decoding and VRAM Layouts for 2-Bit LLM Weights*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
Leach Lattice Vector Quantization (LLVQ) 在 2-bit 量化中展现出最强的质量，但其原始实现仅解码单个格壳（single shell），无法满足完整 2-bit 码本（multi-shell）的解码需求。该码本包含 301 个坐标排列类（coordinate-permutation classes），直接在 GEMV 内部进行组合索引解码会导致严重的 **warp divergence**，从而影响推理效率。

本文首次实现了完整的 **301-class A24(12) 码本** 的高效服务路径，并系统地测量了其在实际部署中的代价。

### 提出的新方法与创新
- **Fused Multi-Shell Decoder (融合多壳解码器)**：
  - 提出一种“离线展开”（offline expansion）策略：在加载时将 47-bit 的组合索引转换为 GPU 友好的布局（如 binary bit planes），避免在 GEMV 中实时解码。
  - 设计了一个 **fused dequantize-plus-matvec kernel**，通过 shifts、masks 和 predicated table lookup 实现无分支发散（divergence-free）的解码。
  - 组合索引本身不在 matvec 中被显式解码，而是以预展开的形式存储于 VRAM。

- **引入“in-VRAM rate”作为独立设计轴**：
  - 区分 **on-disk rate**（磁盘存储率）与 **in-VRAM rate**（运行时读取率），指出后者才是决定模型能否加载的关键。
  - 探索了四种 bit-exact 的 VRAM 布局（SLOT32, PLANES14, PLANES12x, GOLAY70），分析其大小与速度权衡。

- **成本归因分析**：
  - 对比了 LLVQ 与主流 4-bit (AWQ) 和 2-bit (QTIP) GEMV kernel 的性能差距，定位瓶颈在于“必须展开大码本”带来的额外内存流量。
  - 测量了 launch geometry 的开销，并验证融合 q/k/v 投影可提升端到端吞吐。

### 相比现有方法的优势
- 首次公开实现并验证了完整 LLVQ 多壳解码器的服务路径。
- 在相同质量前提下，提供了对高维格码本服务成本的精确建模。
- 揭示了“码本太大无法放入 LUT”所带来的本质性性能代价，而非算法低效。

---

## 2. 核心实验方法和设置

### 数据集与模型
- **主干模型**：`Qwen3-4B`, `Qwen3-8B`, `Qwen3-14B`
- **量化对象**：所有线性层权重（除 embedding 和 output head 外）
- **量化方式**：LLVQ（基于 Leech Lattice A24 的向量量化），码率为 **2.000 b/weight**
- **校准数据**：C4 子集（约 131k tokens）

### 实验设置
- **硬件平台**：
  - 主要测试卡：**NVIDIA L40S**（GDDR6，带宽受限）
  - 对比平台：**NVIDIA A100**（HBM2e）
- **批处理模式**：**batch 1** 的 decode-phase GEMV（即自回归生成场景）
- **流程控制**：所有对比方法运行在同一进程中，采用固定调度顺序，确保公平比较。

### 评估指标
| 指标 | 说明 |
|------|------|
| **Throughput (tok/s)** | 端到端每秒生成 token 数 |
| **VRAM Usage (GB)** | 显存占用（仅权重） |
| **Speedup vs FP16** | 相对于 FP16 matvec 的加速比 |
| **Fraction of byte bound** | 达到理论带宽的比例 |
| **Perplexity (WikiText-2)** | 语言建模能力评估 |
| **MMLU (%)** | 多任务理解能力评估 |

### 基线方法对比
| 方法 | 类型 | 来源 |
|------|------|------|
| **FP16 (control)** | 全精度基准 | 自研 kernel |
| **AWQ w4g128** | 4-bit 逐激活感知量化 | 官方实现移植 |
| **QTIP 2-bit** | 2-bit Trellis 编码 | 官方 inference kernel |
| **cuBLAS FP16** | 工业级 FP16 GEMM | NVIDIA cuBLAS |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（L40S 上 Qwen3-4B）

#### VRAM 布局性能对比（Table 1 & Fig. 3）

| Layout | b/weight (kernel) | Speedup vs FP16 | Fraction of byte bound |
|--------|-------------------|------------------|-------------------------|
| **FP16 (control)** | 16.000 | 1.00× | — |
| **SLOT32** | 5.510 | 1.89× | 65% |
| **PLANES14 (ours)** | 4.804 | **2.15×** | **65%** ✅ |
| **PLANES12x** | 4.342 | 2.00× | 54% |
| **GOLAY70** | 3.589 | 1.34× | 30% |
| **AWQ w4g128 (4-bit)** | 4.179 | 3.38× | 88% |
| **QTIP 2-bit (comp.)** | 2.000 | **4.89×** | **61%** |

> 📌 **PLANES14 是本文选定的“served layout”**，兼顾速度与紧凑性。

#### 与部署级 kernel 的直接对比（C3）
- QTIP kernel 比 PLANES14 **快 2.27×**
- QTIP 读取 **0.91 GB/token**，而 PLANES14 读取 **2.18 GB/token**（**2.40× 更多字节**）
- 效率相近（61% vs 65% of byte bound）→ 时间差 ≈ 字节差 → **代价来自“展开码本”的内存流量**

#### 端到端吞吐（Table 3）

| Model | Dense (tok/s) | Fused (tok/s) | Gain (same f16 head) |
|-------|---------------|----------------|------------------------|
| **4B** | 43.5 | 48.3 | **1.11×** |
| **8B** | 26.5 | 34.1 | **1.29×** |
| **14B** | 17.0 | 23.9 | **1.41×** |

> ✅ 吞吐增益随模型增大而增加。

#### 显存占用（Table 5）
| Model | Dense VRAM | Served VRAM | Ratio |
|-------|------------|-------------|--------|
| **4B** | 8.04 GB | **2.60 GB** | ×3.1 |
| **8B** | 16.38 GB | **5.45 GB** | ×3.0 |
| **14B** | 29.54 GB | **9.39 GB** | ×3.1 |

> ✅ 显存大幅下降，支持更大模型部署。

#### 质量损失（Table 5）
- **4B 模型**：
  - Perplexity: ×1.384（从 12.24 → 16.94）
  - MMLU: 下降 14.7 pts（70.3 → 55.6）
- **缺陷集中在推理类科目**（如抽象代数、会计学降至随机水平）

#### 消融实验结果
- **Launch Fusion (q/k/v + gate/up)**：
  - 减少 kernel launch 从 252 → 144
  - 回收 **11.7% 的 PLANES14 kernel 时间**
  - 端到端增益 **1.061×**
- **A100 平台迁移**：
  - 所有 lattice arms 性能均低于 FP16（因指令发射成为瓶颈）
  - 表明当前优化依赖 GDDR 架构的带宽限制特性

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **首次实现并验证了完整 LLVQ 多壳解码器**，可在 L40S 上达到 **65% byte bound**，证明其工程可行性。
2. ✅ **in-VRAM rate 是独立且关键的设计维度**，PLANES14 在 4.80 b/w 下实现最优平衡。
3. ✅ **性能差距主要源于“码本展开”带来的额外内存流量**，而非 kernel 效率低下；QTIP 因使用 trellis state（可查表）避免此问题。
4. ✅ **端到端吞吐增益随模型规模扩大而提升**（1.11× → 1.41×），表明小模型不是最佳应用场景。
5. ❗ **质量损失显著**，尤其在推理密集型任务上，可能与训练数据分布（C4 vs DCLM-edu）有关。

### 局限性
| 限制项 | 说明 |
|--------|------|
| **硬件依赖性强** | 当前优势仅在 GDDR 类显卡（如 L40S）成立，在 HBM 卡（如 A100）上反被超越 |
| **不支持 batching** | kernel 设计针对 batch 1，未考虑 batched GEMM 场景 |
| **旋转基底共享内存墙** | 输入维度超过一定阈值（如 Qwen3-32B）会超出 shared memory 限制 |
| **加载时展开不可持久化** | 每次启动需重新展开，增加冷启动延迟（4B 加载约 84 秒） |

### 未来工作方向
1. 攻击 **launch geometry 开销** 和 **load-time unfolding 成本**，而非单纯追求相对于 FP16 的加速比。
2. 探索是否能在 **70B 级别模型** 上维持当前趋势（目前仅有 4B/8B/14B 数据）。
3. 验证以下潜在改进方向能否缩小 MMLU 差距：
   - 使用 **reasoning-curated 数据集**（如 DCLM-edu）进行 calibration
   - 引入 **learned per-column scales**
   - 添加 **low-rank compensation**
4. 优化 **shared memory usage**，例如使用 half-precision 激活缓存以突破大模型限制。

---

> 🔚 **总结一句话**：  
> 本文成功构建了首个可部署的 LLVQ 多壳解码器，揭示了高维格码本服务的本质代价是“内存流量”，而非计算效率，并指出未来应聚焦于减少 launch 开销与加载成本，而非盲目追求峰值吞吐。

</details>

---

### 7. [Codebook Agent: Amortized Topology Design for LLM Multi-Agent Systems](https://arxiv.org/abs/2609.02264)

**Authors**: Jinxi Yu, Yubei Li, Eric Hanchen Jiang, Zhi Zhang, Dong Liu, Wenxiao Zhao, Levina Li, Kai-Wei Chang, Ying Nian Wu  
**Category**: cs.AI  
**Published**: 2026-09-03  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.02264v1  

#### Abstract
Adapting the communication topology of an LLM multi-agent system to each query improves both accuracy and efficiency, yet current designers treat this as conditional graph generation: a variational, autoregressive, or diffusion decoder searches the $N \times N$ adjacency space, and a graph-network p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Codebook Agent: Amortized Topology Design for LLM Multi-Agent Systems**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
在基于 **LLM 多智能体系统**（LLM Multi-Agent Systems）中，通信拓扑结构（communication topology）对任务准确性和推理效率有显著影响。传统方法通过**条件图生成**（conditional graph generation）为每个查询动态设计拓扑，例如使用变分、自回归或扩散模型搜索 $N \times N$ 邻接矩阵空间，并依赖图网络代理（graph-network proxy）对候选拓扑进行评分。

然而，作者指出这一范式存在三个根本性缺陷：
1. **有效拓扑空间极小**：尽管搜索空间巨大，但实际“成功”的拓扑仅集中在约6种不同结构上。
2. **结构代价误导**：常用的稀疏性正则项（如边数 $|E|$）与真实 token 消耗负相关（Pearson $r \sim -0.4$），即更稀疏的图反而导致更长的推理链和更高的 token 成本。
3. **消息传递评分器失效**：在主流的同质团队（homogeneous teams）中，由于所有 agent profile 相同，消息传递网络（MPNN/GNN）无法区分不同拓扑，输出恒定。

因此，当前方法在计算上昂贵且逻辑错位。

---

### **提出了什么新方法或新思路**
提出 **Codebook Agent** ——一种**摊销化**（amortized）的拓扑设计框架，其核心思想是将拓扑设计从“生成-评分”转变为“索引-选择”。

#### **三大组件**：
1. **Vector-Quantized Autoencoder（VQ-VAE）**  
   - 在离线阶段压缩历史成功拓扑（$u > 0.5$）为一个**查询无关的 codebook**（16个条目）。
   - 每个 code 对应一个固定的二值邻接矩阵 $A^{(k)}$。

2. **Reward-Weighted MLP Code Predictor**  
   - 将查询嵌入 $c = E(q)$ 映射为 code 上的概率分布 $p_\theta(k|c)$。
   - 训练目标为软标签交叉熵，标签由历史奖励 $R = u - \lambda \hat{T}$ 加权。

3. **Execution-Grounded MLP Proxy**  
   - 接收展平的邻接矩阵 $\text{vec}(A)$ 和查询嵌入 $c$，直接回归**实测效用**（utility）和**归一化 token 成本** $\hat{T}$。
   - 替代传统的 GNN 评分器，避免消息传递开销。

#### **测试时流程**：
- 输入查询 $q$ → 编码为 $c$
- 预测 top-$M$ codes ($M=5$)
- 解码为候选拓扑集合
- 批量前向调用 MLP proxy 得到 $(u, \hat{T})$
- 选择最大化 $u - \lambda \hat{T}$ 的拓扑执行

> ✅ **无采样循环、无消息传递、无迭代优化**

---

### **相比现有方法的优势**
| 维度 | Codebook Agent | Prior Methods |
|------|----------------|---------------|
| **准确性** | 最高（平均 84.6） | 次之（最高 83.0） |
| **延迟** | **2.4 ms** | 301–396 ms（迭代方法） |
| **token 消耗** | ↓ **21.9–33.2%** | 更高（尤其因稀疏图导致冗余通信） |
| **可扩展性** | 固定大小 codebook，不随 $N$ 增长 | 图生成复杂度 $O(N^2)$ |
| **通用性** | 支持异构/同质团队，跨 LLM 后端迁移良好 | 在同质团队中评分器失效 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 类型 | 数据集 |
|------|--------|
| **数学推理** | GSM8K, MATH, MultiArith, SVAMP |
| **代码生成** | MBPP, HumanEval（Pass@1） |
| **综合理解** | MMLU（用于跨后端迁移测试） |

---

### **实验设置**
- **Agent 团队配置**：
  - 数学任务：4× MathSolver + FinalRefer
  - 编程任务：4× Programming Expert + FinalRefer
  - MMLU：3× Knowledgeable Expert
- **LLM 后端**：
  - 主实验：`gpt-4o-mini`（temperature=0.7, max_tokens=1024）
  - 迁移实验：`Qwen-3-8B`
- **Embedding 模型**：`all-MiniLM-L6-v2`（$d=384$），冻结
- **训练数据**：每 benchmark 使用 50 个训练任务 × 6 种固定拓扑（完全图、链、星形、3 个 Erdős–Rényi 样本），共 300 条执行记录 $(A, c, u, T)$

---

### **评估指标**
| 指标 | 描述 |
|------|------|
| **Accuracy / Pass@1** | 主要性能指标 |
| **Median Topology Generation Latency** | 生成拓扑的中位延迟（GPU 单次调用） |
| **Mean LLM Tokens per Query** | 实际消耗的 token 数量 |
| **End-to-End Wall Clock Time** | 完整推理时间 |
| **Pareto Frontier** | 准确率 vs. 延迟 / token 成本的权衡曲线 |

---

### **基线方法对比**
分为三类：
1. **Single-Agent Prompting**  
   - Vanilla, CoT, SC-CoT
2. **Multi-Agent Collaboration**  
   - LLM-Debate, LLM-Blender, DyLAN
3. **Learned Topology Designers**（重点对比对象）
   - GPTSwarm, ADAS, AFLOW, MaAS, AgentDropout
   - G-Designer, ARG-Designer, TopoDIM, GTD（最新迭代/扩散方法）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table 1 & Table 2）**

#### **主实验结果（gpt-4o-mini）**
| Method | Average Accuracy | Best? |
|--------|------------------|-------|
| Vanilla | 77.63 | ❌ |
| DyLAN | 80.18 | ❌ |
| GTD（最强 prior） | 83.02 | ❌ |
| **Codebook Agent (Ours)** | **84.62** | ✅ |

> 在所有六个 benchmark 上均排名第一，平均高出 GTD **+1.60 pts**

#### **跨后端迁移（Qwen-3-8B）**
| Method | Average Accuracy |
|--------|------------------|
| CoT | 67.5 |
| G-Designer | 72.1 |
| GTD | 72.7 |
| **Ours** | **74.0** |

> 表明性能优势非 `gpt-4o-mini` 特有，具有泛化能力

---

### **与基线方法的对比结果**

| 维度 | 结果 |
|------|------|
| **准确性** | 超越所有 baseline，在 MBPP 上提升高达 **+11.1 pts** |
| **延迟** | **2.4 ms** vs. 迭代方法 **301–396 ms** → **快 125–158×** |
| **token 消耗** | 平均减少 **21.9–33.2%**<br>例：GSM8K 从 1239 → 927，MATH 从 2304 → 1611 |
| **端到端时间** | MATH: 43.7 → 27.4 分钟；MMLU: 12.7 → 7.9 分钟 |

> 设计阶段耗时占比降至 **<0.1%**

---

### **消融实验结果（Ablation Studies）**

#### **(a) 团队规模 $N$ 影响（Fig 5a）**
- GSM8K：$N=2$ 到 $10$，准确率稳定在 93–95.5%
- HumanEval：随 $N$ 增加准确率上升（71.9 → 87.5）
- ⇒ 方法适用于不同规模团队，无需架构修改

#### **(b) Codebook Size $K$（Fig 5b）**
- $K$ 从 4 到 64，准确率变化 ≤1.5（GSM8K）/2.5（HumanEval）
- 当 $K > 8$ 时，编码器最多只使用 **6 个 code**
- ⇒ 有效拓扑空间确实很小，额外容量无用

#### **(c) 重排序信号比较（Fig 5c）**
| Scorer | GSM8K Tokens | HumanEval Tokens |
|--------|--------------|------------------|
| MLP（ours） | **927** | **546** |
| Random | 1249 | 750 |
| GNN（incumbent） | 1711 | 918 |

> GNN 比随机还差，因其偏好稀疏图（$|E|$ 小），而稀疏图更耗 token

#### **(d) 成本权重 $\lambda_{\text{cost}}$（Fig 5d）**
- $\lambda = 0$：成本最高（1108 / 743）
- $\lambda = -0.4$：成本最低（927 / 442），但准确率下降
- 默认 $\lambda = -0.1$ 是精度与成本的最佳平衡点

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **有效拓扑空间极小**：无论模型容量如何增长，reward-surviving 拓扑始终收敛至约 **6 种模式**。
2. ✅ **边数不是成本代理**：$|E|$ 与实测 token 消耗呈 **负相关**（$r \sim -0.4$），稀疏图更贵。
3. ✅ **GNN 评分器在同质团队中失效**：因 profile 相同，MPNN 输出与拓扑无关，无法评分。
4. ✅ **摊销化选择优于生成**：通过 codebook 索引 + MLP 评分，可在 **2.4ms 内完成高质量选择**，且节省大量 token。

---

### **方法的局限性**
1. **依赖离线执行数据**：需要预先收集真实 LLM 执行记录 $(A, c, u, T)$ 来训练 codebook 和 proxy。
2. **codebook 容量固定**：虽然 16 已足够，但在极端新任务下可能需重新训练。
3. **假设拓扑可复用**：隐含前提是某些拓扑模式具有跨查询的通用性，若任务极度异构可能受限。

---

### **未来工作方向**
1. **动态扩展 codebook**：在线学习新拓扑并加入 codebook。
2. **轻量化在线微调**：允许 proxy 或 predictor 在部署后适应新任务分布。
3. **结合 agent 角色学习**：同时优化 agent profile 与 topology 结构。
4. **应用于更大规模系统**：探索 $N > 10$ 场景下的 codebook 泛化能力。

---

> 🔚 **总结一句话**：  
> **Codebook Agent 证明了“少即是多”——与其在巨大的图空间中反复搜索，不如先识别出真正有效的少数拓扑，然后用轻量级模型快速选择它们。这不仅更快、更便宜，而且更准确。**

</details>

---

### 8. [How Do Prompt Variations Affect Energy Consumption in On-Device LLMs?](https://arxiv.org/abs/2609.01798)

**Authors**: Wei Hu, Xiaolong Tu, Dawei Chen, Yitao Chen, Kyungtae Han, Haoxin Wang  
**Category**: cs.CL  
**Published**: 2026-09-03  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.01798v1  

#### Abstract
Large language models (LLMs) are increasingly deployed on mobile devices, making energy efficiency a key deployment constraint, yet the energy impact of prompt design remains underexplored. This paper aims to understand how two prompt properties, cognitive load and phrasing pattern, shape the energy...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《How Do Prompt Variations Affect Energy Consumption in On-Device LLMs?》总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**
本文首次系统性地研究了 **prompt 设计** 对 **on-device LLMs**（设备端大语言模型）推理过程中 **能量消耗** 的影响。尽管已有大量研究关注模型压缩、量化等 **model-centric** 优化技术，但 prompt 作为输入层面的关键因素，其对能耗的影响长期被忽视。

该研究旨在回答一个根本性问题：  
> *不同的 prompt 变体是否会导致显著不同的能耗？如果是，这种差异是由哪些 prompt 属性驱动的？*

### **提出了什么新方法或新思路**

#### ✅ 新的数据集构建方法
- 构建了一个新的 **cognitive load**（认知负荷）prompt 数据集，通过控制 **intrinsic load**、**extraneous load** 和 **germane load** 三个子属性，在保持任务语义一致的前提下引入推理复杂度变化。
- 采用 **LLM-based scoring** 和 **embedding similarity** 进行验证，确保生成的变体在语义上对齐且属性纯净。

#### ✅ 提出双维度 prompt 属性分析框架
定义并实证分析两个核心 prompt 属性：
- **Phrasing Pattern**：表面语言风格的变化（如礼貌语气、链式思考 Chain-of-Thought、格式化表达等），不改变语义。
- **Cognitive Load**：推理需求的变化，直接影响模型处理任务所需的认知资源。

#### ✅ 细粒度能效分析范式
提出将 **energy consumption** 分解为两个关键指标进行分析：
- `per-token energy`：每 token 处理的能量成本（反映计算效率）
- `token usage`：处理或生成的 token 数量（反映推理长度）

同时区分 **prefill phase** 和 **decode phase** 的能耗，实现更精细的功耗剖析。

### **相比现有方法的优势**
| 方面 | 传统做法 | 本文优势 |
|------|--------|---------|
| 能耗研究视角 | 聚焦于模型架构、量化等硬件/模型级优化 | 引入 **prompt-level 优化** 作为新的、**model-agnostic** 的节能手段 |
| Prompt 分析方式 | 多关注输出质量、鲁棒性 | 首次建立 **prompt → computational behavior → energy consumption** 的因果链条 |
| 实验设计 | 缺乏对 prompt 属性的隔离控制 | 严格控制变量，分离出 **phrasing** 与 **cognitive load** 的独立影响 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

| 类型 | 名称 | 描述 |
|------|------|------|
| **Phrasing Pattern** | CLEF 2025 ELOQUENT Lab Dataset (Sotic & Kamps, 2025) | 包含 7 种语义等价但表述风格不同的 prompt 变体：<br>• Aggressive Tone<br>• Conversational Tone<br>• Chain-of-Thought (CoT)<br>• Formatting Differences<br>• Persona-Based<br>• Polite Tone<br>• Technical/Jargon-Heavy |
| **Cognitive Load** | 自构数据集 | 基于以下三个公共数据集构造：<br>• **SVAMP**：算术类问答<br>• **BoolQ**：二分类是非题<br>• **AI2-ARC**：科学多选题<br>每个 base prompt 生成三种变体：<br>• Intrinsic Load（分步引导）<br>• Extraneous Load（添加冗余干扰信息）<br>• Germane Load（激活先验知识） |

### **实验设置**

| 项目 | 配置 |
|------|------|
| **LLMs** | 5 个轻量级 on-device 模型：<br>• Gemma-2-2B<br>• Llama-3.2-1B<br>• Qwen-2.5-0.5B / 1.5B<br>• SmolLM2-360M |
| **设备** | Google Pixel 7 和 Pixel 8 Pro |
| **部署框架** | MLC-LLM（支持本地量化推理） |
| **推理配置** | 固定 temperature=0.6, top_p=0.8，最大生成 token 数固定 |
| **功耗测量** | 通过 ADB 采集电压 $V(t)$ 和电流 $I(t)$，计算瞬时功率 $P(t)=I(t)\cdot V(t)$，积分得各阶段能量：<br>$E_{\text{phase}} \approx \sum P(t_i) \Delta t$ |
| **阶段划分** | <br>• **Prefill Phase**：从请求到达至首个 token 输出（含 tokenization、embedding、prefill forward pass）<br>• **Decode Phase**：后续所有 token 生成过程 |

### **评估指标**

| 指标类别 | 具体指标 |
|----------|-----------|
| **能效指标** | • 总能耗（J）<br>• Prefill/Decode 阶段能耗<br>• Per-token energy（J/token）<br>• Token usage（输入/输出 token 数） |
| **响应质量** | • **Cognitive Load 任务**：Exact Match (EM) 准确率<br>• **Phrasing Pattern 任务**：使用 DeepEval 评估六个维度：<br> – Relevance, Correctness, Coherence,<br> – Completeness, Instruction Adherence, Internal Consistency |
| **综合权衡** | Energy-Quality Pareto Frontier 分析 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据与发现**

#### 🔹 RQ1: Per-token energy 主要受什么影响？

| 发现 | 结果说明 |
|------|----------|
| ✅ **绝对 per-token energy 主要由模型决定** | 不同模型间的差异远大于同一模型下不同 prompt 的差异。例如，Gemma-2-2B 的 decode 能耗显著低于其他模型。 |
| ✅ **Cognitive Load 显著提升 decode 阶段 per-token energy** | 在 Qwen-2.5-1.5B 上，extraneous load 导致 decode per-token energy 达到 base prompt 的 **1.29×**。<br>→ 表明高认知负荷增加了解码时每步的计算负担。 |
| ❌ **Phrasing Pattern 对 per-token energy 影响极小** | 所有变体的 normalized per-token energy 接近 1.0，波动小于 10%。<br>→ 表面改写不影响单位计算能耗。 |

#### 🔹 RQ2: Token usage 如何受 prompt 影响？

| 发现 | 结果说明 |
|------|----------|
| ✅ **Phrasing Pattern 显著改变 token usage** | • CoT 和 Aggressive prompts 明显增加总 token 数（最多达 1.8×）<br>• Format 和 Polite prompts 则可能减少 token 数 |
| ✅ **影响具有 model-dependent 特性** | 同一 phrasing pattern 在不同模型上 token 扩展程度不同。例如，CoT 在 SmolLM2-360M 上引发更大增长。 |
| ✅ **不同 phrasing 影响不同推理阶段** | • CoT 和 Aggressive 主要增加 **decode burden**（生成更长回复）<br>• Format prompts 增加 **prefill burden**（解析结构耗时） |

#### 🔹 RQ3: Prompt 是否改变 energy-quality 权衡边界？

| 发现 | 结果说明 |
|------|----------|
| ✅ **某些 phrasing pattern 可同时降低能耗并提高质量** | 在 LLaMA-3.2-1B、Qwen-2.5-1.5B 等模型上，**Format** 和 **Conversational** prompts 位于 Pareto 前沿，优于 base prompt。 |
| ✅ **energy-quality trade-off 是 model-dependent 的** | 同一种 prompt 改写在一个模型上有益，在另一个模型上可能有害。例如，Technical prompt 在 Gemma-2-2B 上略降能耗但在 Qwen 上反而升耗。 |
| ✅ **Cognitive Load 普遍导致“高能耗 + 低准确率”双重惩罚** | 尤其是 **extraneous load**，不仅大幅增加能耗（因输入变长 + 解码变慢），还显著降低 accuracy（干扰信息误导模型）。 |

### **消融实验结果（隐含在设计中）**

- **Single-property isolation**：通过评分过滤机制（target ≥8, non-target ≤3 on 1-10 scale）确保 cognitive load 子属性无交叉污染。
- **Semantic consistency filtering**：要求 prompt 变体与 base 的 embedding cosine similarity ≥ 0.6，防止语义漂移。
- **Length control relaxation**：允许 extraneous/intrinsic prompts 更长，以真实反映其信息密度特性。

---

## 4. 关键结论和发现

### **主要发现**

| 结论 | 说明 |
|------|------|
| 📌 **Cognitive Load 主要影响 per-token energy** | 高推理复杂度会显著增加 decode 阶段每 token 的能耗，尤其在中小规模模型上更为明显。 |
| 📌 **Phrasing Pattern 主要影响 token usage** | 表面语言风格虽不改变单位能耗，但可通过诱导模型生成更长或更短的响应来间接调控总能耗。 |
| 📌 **Prompt 设计可重塑 energy-quality frontier** | 并非所有 prompt 都“平等”，精心设计的 prompt（如 Format、CoT）可在不牺牲甚至提升质量的同时节省能耗。 |
| 📌 **节能效果高度依赖模型架构（model-aware）** | 最优 prompt 策略不能跨模型泛化，必须针对具体模型定制化选择。 |

> 💡 **核心洞见**：  
> Prompt 不仅是“如何提问”的艺术，更是 **energy-aware LLM deployment** 中一个可操作的优化杠杆。未来的 on-device LLM 系统应具备 **prompt-aware energy management** 能力。

### **方法的局限性**

| 局限性 | 说明 |
|--------|------|
| **Prompt 属性覆盖有限** | 仅研究了 phrasing pattern 和 cognitive load，未涵盖情感倾向、对抗性提示等其他重要属性。 |
| **无法完全解耦长度与认知负荷** | Intrinsic 和 Extraneous load 必然导致 prompt 更长，难以完全剥离“长度效应”。 |
| **依赖单一 LLM 进行评估（Gemini-2.5-Pro）** | 可能引入 judge bias，未来需多 judge ensemble 或人工校验。 |
| **硬件平台有限** | 仅在 Pixel 手机测试，不同 SoC/NPU 架构可能表现不同。 |
| **缺乏底层硬件计数器分析** | 当前为端到端测量，未深入 KV-cache、内存带宽等微观机制解释能耗差异原因。 |

### **未来工作方向**

1. **扩展 prompt taxonomy**：纳入更多语义与结构属性（如情绪、讽刺、多模态指令）进行全面能耗映射。
2. **开发 length-matched prompt variants**：通过重写而非增删实现相同长度下的 cognitive load 控制。
3. **构建 prompt-energy prediction model**：基于 prompt 特征预测其能耗行为，用于 runtime 动态调度。
4. **探索 cross-platform profiling**：在更多边缘设备（IoT、车载、AR/VR）上验证结论普适性。
5. **集成 fine-grained tracing tools**：结合硬件性能计数器（perf, ftrace）揭示能耗背后的 micro-architectural 原因。

---

> 🔗 **代码、数据集与脚本公开地址**：[https://amai-gsu.github.io/PromptProperty/](https://amai-gsu.github.io/PromptProperty/)

</details>

---

### 9. [Debias-SparseGPT: Bias-Aware Pruning for Large Language Models](https://arxiv.org/abs/2609.02496)

**Authors**: Irina Proskurina, Guillaume Metzler, Antoine Gourru, Julien Velcin  
**Category**: cs.CL  
**Published**: 2026-09-03  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.02496v1  

#### Abstract
Model compression techniques such as pruning and quantization facilitate the efficient deployment and acceleration of Large Language Models (LLMs). However, recent studies show that weight sparsification methods, such as SparseGPT, can amplify existing biases in models, with outputs varying signific...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Debias-SparseGPT: Bias-Aware Pruning for Large Language Models**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
- 当前主流的模型压缩技术（如 **pruning** 和 **quantization**）虽然能有效提升 **Large Language Models (LLMs)** 的推理效率，但已有研究表明这些方法会**放大模型中的社会偏见（bias）**。
- 特别是像 **SparseGPT** 这类基于二阶重建的剪枝方法，在移除冗余权重时可能导致模型对不同 persona cues（如性别、种族、宗教等）产生不一致甚至歧视性的输出，从而损害模型的公平性。

### **提出了什么新方法或新思路**
- 本文提出 **Debias-SparseGPT**，一种在剪枝过程中显式引入去偏目标的后训练剪枝方法。
- 核心思想是在原始 **SparseGPT** 的重建目标中增加一个**基于成对输入差异的二阶项（△X△Xᵀ）**，该输入对包含语义上对立的 pro-/anti-stereotypical 文本（例如：“黑人从不听父母的话” vs “白人从不听父母的话”）。
- 由此构建出一个**偏差感知的 Hessian 矩阵**：
  $$
  H = X_0X_0^T + X_1X_1^T + 2\Delta X \Delta X^T
  $$
  其中 $\Delta X = X_0 - X_1$，该项鼓励模型在剪枝后仍保持对两类输入的表示差异不变，防止偏见被放大。

### **相比现有方法的优势**
- **首次将去偏目标直接嵌入剪枝过程**，而非事后微调或提示工程。
- 在保留 **SparseGPT** 高效性和低计算开销的同时，显著降低剪枝引发的偏见。
- 同时保持甚至略微优于基线方法在 **perplexity** 和 **zero-shot accuracy** 上的表现。
- 方法通用性强，适用于多种 LLM 架构，并可与其他压缩技术（如量化）结合。

---

## 2. **核心实验方法和设置**

### **使用的数据集**

| 类型 | 数据集 | 描述 |
|------|--------|------|
| **偏见评估** | **UnQover**, **BBQ**, **CrowS-Pairs (CP)** | 用于衡量生成任务中是否避免做出无依据的刻板印象判断（如“谁更可能失败？”）。 |
| **通用性能评估** | **MMLU**, **HellaSwag** | 测试零样本问答与常识推理能力。 |
| **语言建模质量** | **WikiText-2** | 报告 **Perplexity (PPL)** 以评估语言建模保真度。 |
| **校准数据（calibration data）** | **StereoSet dev set**（4,212 对） | 成对的 pro-/anti-stereotypical 句子，用于构建 bias-aware Hessian。 |
| **增强校准数据** | **UltraChat**（部分实验） | 包含多样化长文本对话，用于缓解高稀疏率下的性能退化。 |

### **实验设置和评估指标**

- **模型范围**：共测试 **9 个 LLMs**，涵盖指令调优模型（如 LLaMA-3.1-8B-IT, Vicuna, Qwen）和基础模型（如 Qwen-3-8B, DeepSeek）。
- **稀疏模式**：
  - **非结构化稀疏**（25%, 50%）
  - **半结构化 N:M 稀疏**（1:4, 2:4）
- **评估指标**：
  - **UnQover / BBQ 准确率**：预测 “Not stated” 的比例越高越好 → 表示拒绝无证据的刻板推断。
  - **CrowS-Pairs 分数（CP↓）**：理想值为 50%，越接近说明无系统性偏向。
  - **Perplexity (PPL↓)**：越低越好。
  - **MMLU / HellaSwag 准确率（↑）**：下游任务性能。
  - **DTO (Distance-to-Optimum) ↓**：综合公平性与性能的联合指标，越小越好。

### **基线方法对比**
| 方法 | 类型 | 是否使用输入数据 | 是否考虑偏见 |
|------|------|------------------|--------------|
| **Magnitude Pruning** | 幅值剪枝 | ❌ | ❌ |
| **Wanda** | 激活×幅值评分 | ✅ | ❌ |
| **SparseGPT** | 二阶重建（OBS 扩展） | ✅ | ❌ |
| **Debias-SparseGPT (Ours)** | 偏差感知二阶重建 | ✅ | ✅ |

所有依赖输入的方法均使用相同的 **StereoSet** 校准数据进行公平比较。

---

## 3. **主要实验结果和性能指标**

### **关键性能数据（以 LLaMA-3.1-8B 为例，1:4 结构化稀疏）**

| Method | PPL↓ | MMLU↑ | UnQover↑ | BBQ↑ | DTO↓ |
|--------|------|-------|----------|------|------|
| Dense | 6.99 | 63.17 | 30.10 | 76.40 | — |
| SparseGPT | 8.17 | 59.11 | 35.60 | 67.10 | 0.539 |
| **Debias-SparseGPT (Ours)** | **8.19** | **59.76** | **60.46** | **70.70** | **0.399** |

> ✅ **UnQover 提升巨大**：从 35.60% → **60.46%**，远超 Wanda 最佳值（41.98%），逼近密集模型水平  
> ✅ **DTO 显著下降**：从 0.539 → **0.399**，表明公平-性能权衡最优  
> ✅ **未牺牲通用性能**：MMLU 轻微提升，PPL 几乎持平

### **跨模型与稀疏度的一致性表现**
- 在所有 9 个模型、多种稀疏模式下，**Debias-SparseGPT 均优于 SparseGPT**：
  - **UnQover 平均提升显著**，尤其在原本偏见严重的模型上（如 LLaMA）。
  - **DTO 持续最低**，证明其在公平-性能 trade-off 上全面领先。
  - **CrowS-Pairs 更接近 50%**，显示无系统性反向偏见。

#### 示例：Qwen-2.5-7B 在 2:4 稀疏下的消融实验（Table 4）

| 方法 | Calibration Data | MMLU↑ | UnQover↑ | DTO↓ |
|------|------------------|--------|-----------|-------|
| SparseGPT | StereoSet | 50.41 | 28.45 | 0.616 |
| SparseGPT | +UltraChat | 53.84 | 42.46 | 0.522 |
| Debias-SparseGPT | StereoSet | 48.16 | 24.94 | 0.645 |
| **Debias-SparseGPT** | **+UltraChat** | **54.17** | **47.26** | **0.494** |

> 🔍 发现：**添加 UltraChat 显著改善高稀疏率下的性能**，且 Debias-SparseGPT 收益更大（+22.32 UnQover vs +14.01 for SparseGPT）

---

## 4. **关键结论和发现**

### **主要发现**
1. ✅ **剪枝确实会放大 LLM 中的偏见**，尤其是在 UnQover 和 BBQ 等敏感 QA 任务上。
2. ✅ **Debias-SparseGPT 能有效缓解这一现象**，通过引入成对输入差异的 Hessian 项，在不损失性能的前提下大幅提升公平性。
3. ✅ **方法具有普适性**：在多个 LLM 家族、不同稀疏模式下均稳定有效。
4. ✅ **无需额外训练阶段**：完全 post-training，保持 SparseGPT 的高效性。
5. ✅ **增强校准数据（如 UltraChat）可进一步提升极限稀疏下的表现**，尤其对 2:4 模式至关重要。

### **方法的局限性**
1. 🚫 **仅限英文单语环境**：所用校准与评估数据均为英语，多语言泛化未知。
2. 🚫 **未覆盖全部安全维度**：主要关注 representational bias，对 toxicity、misinformation 等其他风险评估有限。
3. 🚫 **高约束稀疏（如 2:4）仍导致显著性能下降**：需更丰富的校准数据补偿。
4. 🚫 **缺乏对稀疏模式本身的深入分析**：虽发现 `self_attn.o_proj` 层受影响最大（见 Table 17），但尚未解释其机制。

### **未来工作方向**
- 探索多语言版本的 Debias-SparseGPT。
- 将该思想扩展至其他压缩范式（如 **quantization**, **distillation**）。
- 设计理论边界，研究 Hessian 估计对校准数据大小与多样性的敏感性。
- 结合轻量级微调（如 PEFT）进一步优化极端稀疏下的性能。
- 开发更细粒度的偏见控制机制，实现按群体定制的公平性调节。

---

> 💡 **总结一句话**：  
> **Debias-SparseGPT 是首个将去偏目标融入剪枝重建过程的方法，在几乎不增加计算成本的前提下，显著降低了 LLM 压缩带来的偏见放大问题，实现了更优的 fairness-performance trade-off。**

</details>

---

### 10. [H3DNAS: Hardware-Aware ONNX-Native 3D Point Cloud Model Compression](https://arxiv.org/abs/2609.02684)

**Authors**: Anchit Mulye, Rhythm Baghel, Sujay Kumar Ingle, Hardik Jain  
**Category**: cs.LG  
**Published**: 2026-09-03  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.02684v1  

#### Abstract
Deploying 3D point cloud models on edge hardware such as the NVIDIA Jetson Orin Nano is severely constrained by compute and memory budgets. Existing compression methods require access to the model's original source code, rendering them inapplicable to the Open Neural Network Exchange (ONNX) binaries...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# H3DNAS: Hardware-Aware ONNX-Native 3D Point Cloud Model Compression 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前在边缘设备（如 NVIDIA Jetson Orin Nano）上部署 **3D point cloud 模型**面临严重的计算和内存限制。现有的模型压缩方法（如 NAS 或 structured pruning）大多依赖于原始模型的 **source code、训练框架（如 PyTorch）、可微分结构或梯度访问**，这使得它们无法处理由厂商或模型仓库直接发布的 **ONNX 二进制文件**。

因此，当只有 ONNX 模型可用而无源码时，现有方法完全失效。

### 提出了什么新方法或新思路
本文提出 **H3DNAS**，首个无需源码、直接对 ONNX 计算图进行硬件感知压缩的框架，其三大核心贡献如下：

#### (1) Channel Dependency Graph (CDG) 定理
- 将 ONNX 算子分为四类：**Channel-Generating (CG)**、**Channel-Transparent (CT)**、**Channel-Constraining (CC)**、**Channel-Terminating (CX)**。
- 形式化定义了五个约束规则（R1–R5），用于识别不可剪枝的节点。
- **证明自由参数比例 $ p_f $ 是图拓扑不变量**，可在 $ O(|V| + |E|) $ 时间内计算，为压缩提供理论上限（compression ceiling）。
- 可在搜索前快速判断某模型是否可通过通道剪枝满足硬件预算。

#### (2) Two-Stage Hierarchical Search
- **Stage 1**: 基于 L1-importance 的通道剪枝，结合 **output fidelity**（logits 层余弦相似度）作为 zero-shot、无标签的质量代理指标，筛选候选架构。
- **Stage 2**: 对 Pareto 最优候选应用 **GhostConv 结构突变**，进一步突破通道宽度搜索的压缩边界。
- 整个过程不需训练代码，仅通过 **ONNX graph surgery** 实现。

#### (3) 首个 source-code-free 的 3D point cloud 模型压缩流水线
- 完全基于 `onnx.ModelProto` 操作，支持任意来源的 ONNX 模型。
- 支持 fine-tuning 时使用 `onnx2torch` 自动重建可训练模块，无需原始架构定义。
- 内置 **HardwareConstraints**，可针对参数量、FLOPs、模型大小、延迟等硬性指标进行可行性检查。

### 相比现有方法的优势
| 维度 | H3DNAS | 现有方法（如 CP3、DepGraph、AMC） |
|------|--------|-------------------------------|
| **是否需要源码** | ❌ 不需要 | ✅ 必须提供 PyTorch/TensorFlow 源码 |
| **是否支持 ONNX 二进制输入** | ✅ 支持 | ❌ 不支持 |
| **是否支持 zero-shot 评估** | ✅ 使用 output fidelity | ❌ 多依赖 SWAP-Score 或激活统计，高剪枝率下不稳定 |
| **是否集成硬件约束** | ✅ 支持 Jetson 等边缘平台自动优化 | ⚠️ 多数未显式建模 |
| **是否支持 GhostConv 搜索** | ✅ 原生实现 ONNX 级 GhostConv 替换 | ❌ 未见报道 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **ModelNet40**：标准 3D 分类数据集，包含 9,843 个训练样本和 2,468 个测试样本，共 40 类物体，每点云含 1,024 个点。
- 所有实验均在此数据集上进行评估。

### 实验设置和评估指标
- **目标硬件**：NVIDIA Jetson Orin Nano 8GB
  - 预算约束：≤4B FLOPs, ≤20M 参数, ≤50MB 模型大小, ≤50ms 延迟（P50）
- **推理引擎**：
  - 主要使用 **OnnxRuntime (ORT)** CPU 推理（batch=1, 100 次运行取中位数）
  - 补充验证使用 **TensorRT FP16** 在 Jetson GPU 上的表现
- **评估指标**：
  - Top-1 准确率（fine-tuned 后）
  - 参数减少百分比（Parameter Reduction）
  - FLOPs 下降
  - 模型大小（MB）
  - 推理延迟（ms）及加速比（Speedup）
  - 是否满足 Jetson 四项约束

### 基线方法对比
| 方法 | 描述 | 是否 source-free |
|------|------|----------------|
| **Uniform L1** | 所有可剪枝层统一按 L1 范数剪枝 | ❌ |
| **HRank** | 基于特征图秩排序（CP3 中使用） | ❌ |
| **S Screening F-stat** | 基于伪标签下的 ANOVA F-statistic 排序 | ❌ |
| **CP3** | 当前 SOTA 的 3D 模型压缩方法，需完整 PyTorch 源码 | ❌ |
| **T3DNet** | 知识蒸馏训练小型网络，非原模型压缩 | ❌ |
| **HLS4PC** | 减少输入点数量，不改变模型权重 | ❌ |
| **LTH Unstructured** | Lottery Ticket Hypothesis，非结构化稀疏 | ❌ |

> 所有 baseline 使用相同 fine-tuning 设置以保证公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 4）

| Model | Variant | Accuracy | Δ(pp) | Params | Param ↓ | FLOPs ↓ | Size | Latency (P50) | Speedup |
|-------|---------|----------|--------|--------|----------|-----------|--------|----------------|----------|
| **PointNet** | Base | 90.32% | – | 3.46M | – | – | 13.32MB | 11.98ms | 1.00× |
| | H3DNAS | **90.28%** | -0.04 | **1.19M** | **65.5%** | 63.4% | **4.65MB** | **6.02ms** | **1.99×** |
| **PointNet++ SSG** | Base | 91.90% | – | 1.47M | – | – | 5.70MB | 37.14ms | 1.00× |
| | H3DNAS | **91.98%** | **+0.08** | **0.83M** | **43.2%** | 43.1% | **3.27MB** | **28.86ms** | **1.29×** |
| **PointMLP** | Base | 93.40% | – | 13.20M | – | – | 50.86MB | 347.0ms | 1.00× |
| | H3DNAS | **93.11%** | -0.28 | **6.72M** | **49.1%** | 48.8% | **26.07MB** | **207.8ms** | **1.67×** |

> ✅ 所有压缩模型均从 ONNX 文件出发，无需任何源码。

### 与基线方法对比（Table 8）

| Method | Accuracy (Comp.) | Δ(pp) | Param ↓ | Source-free? |
|--------|------------------|--------|----------|---------------|
| **CP3 + HRank** | 92.95% | +0.15 | 43% | ❌ |
| **H3DNAS (ours)** | **91.98%** | **+0.08** | **43.2%** | ✅ |
| **T3DNet** | ~91.0% | -1.45 | 98%* | ❌ |
| **HLS4PC** | 91.69% | -1.91 | * | ❌ |
| **H3DNAS (PointMLP)** | **93.11%** | **-0.28** | **49.1%** | ✅ |

> 💡 H3DNAS 是唯一能在 **trained PointMLP ONNX 图上直接压缩** 并保留高精度的方法。

### 消融实验结果（Table 7）

| Model | Stage | Accuracy | Δ(pp) | Param ↓ | Speedup | GhostConv |
|-------|-------|----------|--------|----------|----------|------------|
| **PointNet** | Stage 1 | 90.07% | -0.24 | 32.3% | 1.46× | ❌ |
| | Stage 2 | **89.99%** | **-0.32** | 32.3% | **1.80×** | ✅ |
| **PointNet++** | Stage 1 | 91.77% | -0.12 | 33.8% | 2.67× | ❌ |
| | Stage 2 | 91.77% | -0.12 | 34.1% | 2.64× | ✅ |
| **PointMLP** | Stage 1 | 93.11% | -0.28 | 26.8% | 1.24× | ❌ |
| | Stage 2 | 92.83% | -0.57 | 18.6% | 1.20× | ✅ |

> 🔍 发现：
> - **GhostConv 对 PointNet 提升显著**（+0.34× 加速），因其有 12 个符合条件的 Conv 层。
> - **PointMLP 反而受损**，因残差连接引入额外 channel coupling，降低效率。
> - CDG 分析可预测该行为：高 $ p_f $ + 多 free Conv → GhostConv 更有效。

---

## 4. 关键结论和发现

### 主要发现
1. **$ p_f $ 是有效的压缩天花板估计器**
   - CDG 定理可在 <1 秒内分析任意 ONNX 模型，判断其是否适合通道剪枝。
   - 如 MobileNetV2 虽 $ p_f = 45.7\% $，但仅有 2 个自由节点，实际难以压缩，应转向量化。

2. **H3DNAS 是首个真正 source-code-free 的 3D 模型压缩方案**
   - 所有操作基于 `onnx.ModelProto`，适用于跨框架、厂商提供的模型。
   - 成功实现了 PointNet、PointNet++、PointMLP 的 ONNX 级压缩，且性能接近甚至超越依赖源码的方法。

3. **output fidelity 是更鲁棒的 zero-shot 代理**
   - 相比 SWAP-Score，在高压缩比下仍能稳定反映功能一致性。
   - 特别适用于 GhostConv 初始化后的模型评估。

4. **架构特性决定压缩策略选择**
   - **简单 CNN 架构**（如 PointNet）：高 $ p_f $，适合深度剪枝 + GhostConv。
   - **残差密集架构**（如 PointMLP）：建议只用 Stage 1 剪枝，避免 GhostConv 引入复杂耦合。

### 方法的局限性
- **延迟评估基于 ORT CPU**，虽与 GPU 趋势一致（见 Table 14），但缺乏 on-device profiling。
- **暂不支持梯度驱动的 NAS**（如 DARTS），因 ONNX 不保留可微性。
- **Level 2 结构突变**（如 depthwise decomposition）需重新初始化权重，尚未完全支持。
- **onnx2torch 重构存在兼容性问题**：部分自定义算子或导出模式需手动注册。

### 未来工作方向
- 扩展至更多硬件平台（如 FPGA、TPU）并集成更精确的 on-device profiler。
- 支持联合压缩（pruning + quantization）pipeline，如与 ORT-Quantization 集成。
- 探索基于 CDG 的自动 head pruning（尤其适用于 Transformer 类模型如 PTv3）。
- 开发可视化工具，帮助用户理解 CDG 约束分布与压缩潜力。

---

> ✅ **一句话总结**：  
> H3DNAS 首次实现了 **无需源码、纯 ONNX 图操作的 3D point cloud 模型压缩**，通过 CDG 理论指导 + 两级搜索策略，在保持精度的同时显著提升推理速度，并全面适配边缘硬件约束，为工业级模型部署提供了实用解决方案。

</details>

---

### 11. [HeadWiseKV: Budgeted Per-Head Cache Residency for Hybrid Long-Context Language Models](https://arxiv.org/abs/2609.02029)

**Authors**: Renjie Xie, Juncheng Yang, Aoting Hu, Mingxi Zhang, Liyao Wu, Zheheng Hong, Wei Xu  
**Category**: cs.AI  
**Published**: 2026-09-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.02029v1  

#### Abstract
Long-context inference retains a growing key--value (KV) cache during decoding, which consumes substantial GPU memory and can reduce generation throughput. This bottleneck remains in hybrid language models because their residual global-attention layers can dominate context-dependent cache demand. We...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# HeadWiseKV: Budgeted Per-Head Cache Residency for Hybrid Long-Context Language Models 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在**长上下文自回归推理**（long-context autoregressive inference）过程中，Transformer 模型中的 **Key-Value (KV) cache** 随着上下文长度增长而持续膨胀，消耗大量 GPU 显存（VRAM），成为生成吞吐量的瓶颈。尽管**混合语言模型**（hybrid language models）通过引入局部滑动窗口注意力（sliding-window attention）、循环结构（recurrent blocks）等机制缓解了这一问题，但其残余的全局注意力层（residual global-attention layers）仍需存储完整的 KV 缓存，导致显存需求依然随上下文线性增长。

现有方法可分为两类：
- **动态策略**（prompt-dependent）：基于当前请求内容在线决定保留哪些 token 或 head，虽能适应特定输入，但带来运行时开销，且无法降低预填充阶段（prefill）的峰值显存。
- **静态策略**（static）：如 StreamingLLM、RazorAttention 等，采用固定规则（如 sink + recent window），部署简单但粒度粗（如二元划分 retrieval vs. streaming heads），难以适配不同层、不同 head 对历史依赖的异质性。

### 提出的新方法与创新思路
本文提出 **HeadWiseKV**，一种无需训练的框架，用于在混合长上下文模型中实现**预算化的每头缓存驻留**（budgeted per-head cache residency）。其核心思想是：

- **多级静态后缀窗口分配**：为每个可配置的全局 KV head 分配一个固定的、连续的历史窗口长度（如 8K, 16K, 32K, full），形成一个“多级”而非“二元”的压缩策略。
- **前缀条件化校准算法 SeqCalib**：离线执行，逐层处理，在已确定的下层策略基础上，评估候选窗口对注意力输出的影响，选择满足相似性阈值的最小成本窗口。这确保了高层决策是在实际部署的低层截断状态下做出的，提升了策略一致性。
- **分组物理缓存运行时**（grouped-cache runtime）：直接按策略分配物理内存，仅存储选定长度的 KV 状态，而非在全尺寸缓存上应用逻辑掩码。这使得**缓存策略直接决定物理显存占用**，实现真正的显存节省。

### 相比现有方法的优势
| 维度 | HeadWiseKV | 其他静态方法（如 StreamingLLM, DuoAttention） | 动态方法（如 AdaKV, HeadKV-R2） |
|------|------------|---------------------------------------------|-------------------------------|
| **策略时机** | 静态（offline） | 静态 | 动态（online） |
| **分配粒度** | 多级 per-head | 通常为二元或单层统一 | 可细粒度，但依赖输入 |
| **物理驻留** | 是（直接分配） | 是（部分） | 否（常先建全缓存再压缩） |
| **部署一致性** | 是（prefix-conditioned） | 否（假设全历史） | 是（但运行时开销大） |
| **预填充峰值显存** | 显著降低 | 可降低 | 不降低（因先建全缓存） |

> ✅ **优势总结**：HeadWiseKV 在保持部署简单性和低运行时开销的同时，实现了更精细、更一致的缓存压缩，真正降低了物理显存占用，并支持更长上下文。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **校准数据集**（Calibration Dataset）：
  - **WikiText-103**：用于离线运行 SeqCalib 算法，生成缓存策略。
- **下游任务评估数据集**：
  - **RULER**：评估长上下文检索能力（needle-in-a-haystack），涵盖 5 个任务，上下文长度从 4K 到 128K。
  - **LoCoMo**：评估对话记忆能力，共 1,540 个问题，使用 DeepSeek V4 Flash 作为评判模型。
  - **T2-Bench**：评估代理任务表现，包含 Airline 和 Retail 两个领域。
  - **LongMemEval-S**（补充材料）：额外的长期记忆评测集。

### 实验设置和评估指标
- **模型**：
  - 主要研究模型：**Qwen3.6-27B**（Q4_K_M 量化，F16 KV）
  - 跨模型验证：Qwen3.5-9B, Qwen3.6-35B-A3B, Gemma4-31B
- **上下文长度**：8K, 16K, ..., 128K, 最高测试至 161K
- **硬件平台**：RTX 4090 D
- **评估指标**：
  - **质量指标**：RULER 得分（0–100）、LoCoMo 正确率、T2-Bench 任务成功率。
  - **系统指标**：
    - 采样峰值显存（sampled peak VRAM）
    - 预填充吞吐量（prefill throughput, tok/s）
    - 解码吞吐量（decode throughput, tok/s）
    - 最大成功上下文长度（verified context capacity）

### 基线方法对比
| 基线方法 | 类型 | 特点 |
|---------|------|------|
| **Full-KV** | 上限基准 | 保留所有 KV 缓存 |
| **AdaKV** (Feng et al. 2025) | 动态 | 基于注意力分数动态分配 head 级预算 |
| **HeadKV-R2** (Fu et al. 2025) | 动态 | head 级压缩，集成检索与推理路径 |
| **StreamingLLM** (Xiao et al. 2024b) | 静态 | 固定 sink + recent window（4 + 90,364） |
| **DuoAttention** (Xiao et al. 2025) | 静态 | 二元划分 retrieval / streaming heads |

> ⚠️ 注意：仅 Full-KV 与 HeadWiseKV 属于“匹配队列”（matched cohort），用于公平比较显存与吞吐；其他基线为独立测量，报告绝对值。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 在 Qwen3.6-27B 上的结果（T = 0.995，保留率 68.95%）
| 指标 | HeadWiseKV | Full-KV | 提升/变化 |
|------|-----------|--------|----------|
| **RULER** | 99.65 | 100.00 | ↓0.35（几乎无损） |
| **LoCoMo** | 94.55 | 94.55 | 无变化 |
| **T2-Bench (Airline)** | 79.50 | 77.00 | ↑2.5 |
| **T2-Bench (Retail)** | 82.24 | 82.02 | ↑0.22 |
| **112K 峰值显存** | 20.0 GiB | 22.0 GiB | ↓8.59% |
| **最大成功上下文** | **161K** | 114K | ↑1.41× |
| **128K 解码吞吐** | 34.24 tok/s | OOM | 成功运行 |

#### 跨模型迁移效果（T = 0.995）
| 模型 | RULER (Full → HW) | LoCoMo (Full → HW) | 保留率 |
|------|------------------|------------------|--------|
| Qwen3.6-27B | 100.00 → 99.65 | 94.55 → 94.55 | 68.95% |
| Qwen3.6-35B-A3B | 100.00 → 99.94 | 92.99 → 90.91 | 74.38% |
| Gemma4-31B | 99.96 → 99.50 | 94.87 → 94.94 | 91.89% |
| Qwen3.5-9B | 99.90 → 99.29 | 91.30 → 89.74 | 58.01% |

> 🔍 发现：HeadWiseKV 在所有模型上均接近 Full-KV 表现，尤其在 RULER 上几乎无损。

### 与基线方法的对比结果（128K 上）
| 方法 | RULER | LoCoMo | 峰值显存 | 解码吞吐 | 成功运行？ |
|------|-------|--------|----------|----------|------------|
| Full-KV | 100.00 | 94.55 | OOM | OOM | ❌ (115K OOM) |
| AdaKV | 92.83 | 94.22 | OOM | — | ❌ |
| HeadKV-R2 | 39.65 | 94.22 | OOM | OOM | ❌ |
| StreamingLLM | 83.88 | 82.40 | 20.0 GiB | 32.81 | ✅ |
| DuoAttention | 88.58 | 82.60 | 18.0 GiB | 32.16 | ✅ |
| **HeadWiseKV** | **99.65** | **94.55** | **20.0 GiB** | **34.24** | ✅ (161K) |

> ✅ HeadWiseKV 是唯一在**质量几乎无损**的前提下，显著**扩展上下文容量**并**维持高解码吞吐**的方法。

### 消融实验结果
- **不同保真度阈值 $ \tau $ 的影响**（见 Table 3）：
  - 当保留率从 100% 降至 85.74%（$ \tau = 0.998 $），RULER 和 LoCoMo 仍保持满分。
  - 即使在最强压缩（68.95%，$ \tau = 0.995 $），RULER 仅下降 0.35 分，LoCoMo 无变化。
  > 💡 表明模型存在**显著的异构冗余**（heterogeneous redundancy），可在不影响性能前提下大幅压缩。

- **跨模型迁移有效性**：
  - 使用相同 $ \tau = 0.995 $ 可在不同模型上生成不同保留率的策略，说明 $ \tau $ 是**保真度阈值**而非固定压缩率。
  - 质量稳定性表明 SeqCalib 具有良好泛化能力，但最终部署仍需在目标 workload 上验证。

---

## 4. 关键结论和发现

### 主要发现
1. **残差全局注意力层是长上下文瓶颈的关键**：即使在混合模型中，少量全局 attention 层仍主导显存需求。
2. **非均匀、多级 per-head 分配优于统一或二元策略**：不同 head 对历史依赖敏感度不同，精细化分配可兼顾效率与性能。
3. **部署一致性校准至关重要**：高层决策必须基于实际部署的低层截断状态进行校准（prefix-conditioned），否则会因表示偏移导致性能下降。
4. **物理缓存驻留是显存优化的根本**：只有在 prefill 前就分配压缩后的物理内存，才能真正降低峰值显存。
5. **HeadWiseKV 实现了质量-效率-容量的最优平衡**：在多个模型上接近 Full-KV 质量，同时降低显存、提升容量、维持高吞吐。

### 方法的局限性
- **静态策略限制**：无法恢复被截断的重要远距离信息，不适用于具有非常规长程依赖的请求。
- **需重新校准**：当模型版本、KV 格式、目标上下文长度或目标任务发生变化时，需重新运行 SeqCalib。
- **非全局最优**：SeqCalib 是逐层贪心搜索，不能保证联合全局最优。
- **依赖实现细节**：需访问 per-head Q/K/V 张量和 GQA 映射关系，对推理引擎有一定要求。

### 未来工作方向
- 探索轻量级动态机制与 HeadWiseKV 的结合，在关键 head 上实现 selective recovery。
- 将 SeqCalib 扩展至 layer-wise 或 block-wise 更大粒度的联合优化。
- 研究跨任务、跨领域的通用校准策略，减少 per-model 校准成本。
- 结合 KV quantization 与 HeadWiseKV，进一步压缩显存占用。

---

> 📌 **总体评价**：  
> **HeadWiseKV 是一项实用性强、效果显著的系统优化工作**。它没有追求复杂的在线决策，而是通过精心设计的离线校准与物理内存管理，在无需微调的前提下，实现了长上下文推理中**质量、显存、吞吐、容量**的全面提升，为大规模部署混合架构 LLM 提供了可靠的技术路径。

</details>

---

### 12. [MASkills: Continual Skills Optimization for Multi-Agent LLM Systems](https://arxiv.org/abs/2609.02094)

**Authors**: Huaiyuan Yao, Xiaoou Liu, Charles Fleming, Tianlong Chen, Hua Wei  
**Category**: cs.AI  
**Published**: 2026-09-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.02094v1  

#### Abstract
LLM-based multi-agent systems have shown strong performance on complex tasks, yet continual improvement from interaction experience remains challenging. Existing self-reflection methods build experience memories, but memories are mostly hard to invoke, refine, or scale, while agent skills offer a mo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MASkills: Continual Skills Optimization for Multi-Agent LLM Systems

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的基于 LLM 的多智能体系统（multi-agent LLM systems）虽然在复杂任务中表现出色，但在**持续学习与经验优化**方面仍面临挑战。传统方法依赖 **self-reflection** 构建 experience memory，但这类记忆存在以下问题：
- 难以调用、更新和扩展；
- 缺乏明确的触发条件和可操作性；
- 随着时间推移，记忆变得冗余、噪声大且难以规模化。

此外，在多智能体环境中，如何将团队奖励归因到具体的 **agent** 和其使用的 **skill** 上，即实现细粒度的信用分配（credit assignment），是一个尚未有效解决的问题。

### 提出了什么新方法或新思路
本文提出 **MASkills** —— 一种面向多智能体系统的**持续技能优化框架**，其核心思想是：
> 将策略改进（policy improvement）视为对 **skill space** 的优化，而非传统的参数空间优化。

#### 创新点包括：
1. **Skill 作为可复用的知识单元**  
   引入结构化的 **skill artifacts**（如 `SKILL.md` 文件、配置文件、脚本等），封装“何时用、怎么用、用什么工具”的过程性知识，支持轻量检索、按需加载和渐进式披露（progressive disclosure）。

2. **Skill-conditioned Credit Assignment**  
   设计了一个语言型 critic 模块，通过反事实分析（counterfactual comparison）为每个 skill invocation 分配信用信号，判断该 skill 是否促进或阻碍了最终团队目标。

3. **Hierarchical Credit Aggregation + Momentum-Smoothed Optimization**  
   在多个维度（trajectories, agents, skills, topologies）上聚合语言反馈，并引入 momentum 机制平滑历史编辑方向，提升优化稳定性。

4. **Credit-Driven Skill Evolution Pipeline**  
   定义四个核心操作符驱动 skill 库的演化：
   - **Refinement**：局部修改已有 skill；
   - **Induction**：从失败案例中生成新 skill；
   - **Consolidation**：合并功能重叠的 skill；
   - **Pruning**：移除低效或过时 skill。

5. **Validation & Rollback 机制**  
   所有 skill 更新必须通过 hold-out validation 验证，否则回滚，防止行为退化，相当于在 skill space 中实现了 **trust region** 控制。

### 相比现有方法的优势
| 方面 | 现有方法（如 Reflexion, TextGrad） | MASkills |
|------|-------------------------------|---------|
| 学习单位 | Free-form memory 或 prompt 修改 | 结构化、可组合的 skill artifacts |
| 反馈粒度 | Agent-level 或 trajectory-level | Skill-level 细粒度 credit assignment |
| 多智能体适配 | 多数为单 agent 设计 | 显式建模 multi-agent 协作动态 |
| 优化方式 | Prompt rewriting / 参数微调 | Language-space pseudo-gradient 更新 |
| 稳定性保障 | 无显式控制 | Momentum smoothing + validation rollback |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在三个代表性 benchmark 上进行：
1. **HotpotQA**：多文档问答任务，要求跨多个支持证据进行 multi-hop 推理。
2. **LoCoMo**：超长对话记忆任务，评估模型在多轮会话中的持久记忆与时间推理能力。
3. **GAIA**：通用 AI 助手基准测试，涵盖推理、多模态理解、网页浏览、工具使用等真实场景任务。

### 实验设置
- **环境建模**：所有任务被形式化为 **cooperative Dec-POMDP**，多个 LLM agent 分工协作完成共享目标。
- **Agent 角色设计**：包括信息检索、验证、规划、记忆追踪、决策等角色，通过去中心化通信交互。
- **Skill 表示**：每个 skill 是一个目录，包含：
  - `skill.yaml`（元数据）
  - `SKILL.md`（指令）
  - `resources/`（辅助资源）
- **Optimizer Backbone**：使用 GPT-4o-mini 进行 credit assignment 和 skill 编辑。
- **Actor Backbone**：执行 agent 使用 GPT-4o-mini（HotpotQA, LoCoMo）或 Qwen2.5-7B（GAIA）。

### 评估指标
| 数据集 | 主要指标 |
|-------|--------|
| HotpotQA | Answer-level F1 |
| LoCoMo | F1 和 BLEU（分 single-hop / multi-hop） |
| GAIA | 平均成功率（Accuracy） |

同时报告 skill quality 和 transferability 分析。

### 基线方法对比
- **MemoryBank**, **ReadAgent**：基于 memory 的 baseline
- **LoCoMo**, **MemGPT**：当前 SOTA 的长期记忆 agent
- **AutoGen**, **MetaGPT**：主流 multi-agent 框架
- **TextGrad**, **DSPy**, **Reflexion**：自优化 prompt 或 memory 方法

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 和 Figure 3）

| 方法 | HotpotQA F1 | LoCoMo-SH F1 | LoCoMo-MH F1 | GAIA Acc |
|------|-------------|--------------|--------------|----------|
| MemoryBank | – | 5.00 | 5.56 | – |
| ReadAgent | – | 9.15 | 5.31 | – |
| LoCoMo | – | 25.02 | 12.04 | – |
| MemGPT | – | 26.65 | 9.15 | – |
| **MASkills** | **76.30** | **27.61** | **17.22** | **23.3** |

> 注：Decentralized Peer topology 下的结果。

#### 性能优势总结：
- 在 **HotpotQA** 上显著优于其他 multi-agent 推理方法，显示更强的 multi-hop 推理与证据整合能力。
- 在 **LoCoMo** 上取得最佳 single-hop 和 multi-hop 表现，说明其在长期记忆协调与时间推理上的优越性。
- 在 **GAIA** 上达到最高准确率，体现其在开放世界任务分解、工具调用和综合决策方面的优势。

### 消融实验结果（Table 2）
消融研究验证了各组件的重要性：

| 方法变体 | LoCoMo-MH F1 | GAIA Acc |
|--------|---------------|-----------|
| **MASkills (Full)** | 17.2 | 23.3 |
| w/o Skill Credit Assignment | 14.2 | 17.1 |
| w/o Momentum Smoothing | 16.4 | 21.9 |
| w/o Validation Rollback | 6.6 | 13.5 |
| w/o Consolidation / Pruning | 13.9 | 13.0 |

> **关键发现**：
> - 移除 **validation rollback** 导致性能崩溃，证明其对稳定性的关键作用；
> - **skill-level credit assignment** 对性能提升贡献最大之一；
> - momentum smoothing 和 consolidation/pruning 提供稳定增益。

### 跨任务技能迁移实验（Figure 3b）
- 将在 GAIA 上训练的 skill 迁移到 HotpotQA，性能超过 CoT 和 MultiPersona；
- 将 LoCoMo 的 skill 用于其他 long-horizon memory 任务，也能带来一致提升；
- 表明 MASkills 学得的是**可泛化的 procedural abstractions**，而非任务特定提示。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Skill 是比 memory 更有效的持续学习单元**：结构化、模块化、可解释的 skill 支持高效复用与迭代优化。
2. ✅ **Skill-level credit assignment 至关重要**：只有将反馈精确归因到具体 skill，才能实现有针对性的策略改进。
3. ✅ **语言空间优化可行且有效**：即使没有梯度，也可通过 LLM-based critic + aggregation + editing 实现稳定的 policy evolution。
4. ✅ **去中心化架构更利于探索类任务**：在 HotpotQA 和 GAIA 中，decentralized peer coordination 表现最好；而 LoCoMo 因需全局一致性，centralized 更优。
5. ✅ **技能具有可迁移性**：学到的 skill 能跨任务迁移并提升性能，表明其捕获了通用的行为模式。

### 方法的局限性
1. **应用场景受限于合作设定**：目前仅适用于 cooperative multi-agent 场景，未考虑 adversarial 或竞争性环境。
2. **角色与拓扑相对固定**：agent roles 和 communication topology 在训练中保持不变，缺乏动态组织适应能力。
3. **技能库膨胀风险**：随着 skill 不断 induction，可能出现检索效率下降、管理复杂等问题。
4. **依赖高质量 LLM optimizer**：整个优化流程依赖强大的 LLM（如 GPT-4o）作为 critic 和 editor，成本较高。

### 未来工作方向
1. **扩展至动态组织结构**：支持 agent 自主调整角色、拓扑或分工。
2. **支持竞争性多智能体游戏**：将 skill optimization 应用于 zero-sum 或 mixed-motive 设置。
3. **大规模技能管理系统**：研究 hierarchical skill organization、retrieval compression 和 lifelong learning 机制。
4. **降低对强 LLM 的依赖**：探索轻量化 critic 或 distillation 方法，使框架更易部署。
5. **增强安全性与可控性**：结合 human-in-the-loop 审核机制，防范 unsafe skill 演化。

---

> 🔗 **代码开源地址**：[https://github.com/DaRL-GenAI/MASkills](https://github.com/DaRL-GenAI/MASkills)

</details>

---

### 13. [SCX Router: Streaming Zero-Shot Model Selection with a Decoder-KV Classifier and a Real-World Task Ontology](https://arxiv.org/abs/2609.02292)

**Authors**: Ihor Stepanov, Aleksandr Smechov, Mykhailo Shtopko, Dmytro Vodianytskyi, Oleksandr Lukashov  
**Category**: cs.AI  
**Published**: 2026-09-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.02292v1  

#### Abstract
The rapid proliferation of large language models (LLMs) and the growing diversity of their applications presents a unique optimization opportunity: selecting the right model for the task, while optimizing for speed, cost, and quality at a per-task level. However, inference endpoints can vary widely ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SCX Router: Streaming Zero-Shot Model Selection with a Decoder-KV Classifier and a Real-World Task Ontology

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
随着 **Large Language Models (LLMs)** 的快速演进，不同模型在 **质量、延迟、成本、上下文长度、工具调用能力** 等方面差异巨大。如何为每个任务动态选择最优模型，成为一个关键挑战。传统方法依赖手动规则、级联策略或生成式路由（如让另一个 LLM 决定路由），存在以下问题：
- 手动规则难以维护；
- 生成式路由引入额外延迟和格式解析开销；
- 缺乏对多维信号（任务类型、难度、推理模式等）的统一建模；
- 对话场景下重复处理历史上下文效率低下。

### 🚀 提出的新方法
作者提出 **SCX Router**，一种基于 **Decoder-KV 架构的轻量级零样本分类器**，用于实时模型选择。其核心创新包括：

#### （1）Streaming Zero-Shot Router 架构
- 基于 **GLiClass** 框架，将模型选择建模为 **zero-shot 多标签分类任务**，而非生成任务。
- 输入为请求文本 + 自然语言形式的候选模型标签（如 `"Gemma 4 31B"`），输出为每个标签的 suitability score。
- 使用 **Qwen3-0.6B** 作为因果解码器主干，结合一个浅层双向 scorer（类似 DeBERTa），实现高效打分。

#### （2）Decoder-KV Cache 机制
- 在多轮对话中，仅对新增的对话内容进行编码，并将其 KV 状态追加到持久缓存中。
- 候选标签作为“临时后缀”输入，其 KV 状态不写回主缓存，避免污染对话状态。
- 支持 **流式分类**，无需每轮重新处理整个对话历史。

#### （3）多信号预测接口（Multi-Signal Interface）
同一个 checkpoint 可同时预测：
- 模型适用性（model suitability）
- 任务类型（task type）
- 难度等级（difficulty）
- 推理模式（reasoning mode）
- 输出长度（output length）
- 安全信号、自定义标签等

#### （4）现实世界任务本体（Real-World Task Ontology）
构建了一个包含：
- **23 个任务家族（families）**
- **115 种任务类型（task types）**
- **345 个可路由子类型（routable subtypes）**
- **30 个领域（domains）**
- **8 个交叉维度**（如推理层级、风险等级、交互模式等）

该本体支持合成 **150,000 个可验证任务** 和 **15,000 个开放任务**，覆盖真实工作流中的工具使用、代码修改、多智能体协作等复杂场景。

#### （5）路由策略与决策分离
- 学习预测模块（predictor）与部署策略（policy）解耦。
- 路由决策需结合硬约束（如隐私、驻留地、工具权限）和软目标（成本、延迟、缓存复用）。

---

### 🔍 相比现有方法的优势

| 维度 | 传统方法 | SCX Router |
|------|--------|-----------|
| **延迟** | 生成式路由需解码 route token | 非生成式前向传播，无 decoding 开销 |
| **灵活性** | 固定输出头或 prompt engineering | 支持动态标签（dynamic labels），无需重训练即可添加新模型 |
| **上下文复用** | Encoder 需重编码全部历史 | Decoder-KV 缓存仅增量更新，高效支持多轮对话 |
| **多任务支持** | 单一任务专用模型 | 一个 checkpoint 支持多种信号预测 |
| **现实适应性** | 基于静态 QA 数据集 | 基于真实工作流结构的任务本体 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 类型 | 名称 | 规模 | 描述 |
|------|------|------|------|
| **Benchmark-derived** | LiveBench、GAIA、SWE-bench、OSWorld 等 | ~105k 条记录 | 来自多个公开基准的训练集提示，经候选模型执行后收集结果 |
| **Synthetic Tasks** | Ontology-driven synthetic tasks | 150,000 verifier-scored + 15,000 judge-scored | 基于任务本体生成，包含上下文、文件、工具接口、评估标准等 |
| **Label Taxonomy** | 自定义任务本体 | 23 families, 115 types, 345 subtypes | 用于任务分类与属性预测 |

> ⚠️ 注意：最终评估使用的 1,500 个任务来自 13 个 benchmark 的混合，分析时选取了其中 1,000 个具有正增益的子集。

---

### 🧪 实验设置与评估指标

#### （1）模型配置
- 主干：`Qwen3-0.6B`（约 6 亿参数）
- 序列长度：最大 4,096 tokens
- Scorer：两层 DeBERTa-style encoder + 共享 MLP
- 训练阶段：
  - 第一阶段：广义 GLiClass 混合任务预训练（524k records）
  - 第二阶段：聚焦路由任务微调（65k records）

#### （2）评估指标

| 任务类型 | 指标 |
|--------|------|
| 多标签模型适用性预测 | Macro F1（阈值 0.5）、Precision、Recall |
| 单标签任务分类 | Macro F1 |
| 端到端路由效果 | Top-1 Score（router vs. fixed/baseline） |
| 成本感知策略 | Utility function $ U(m) = \alpha U_{\text{perf}} + (1-\alpha) U_{\text{cost}} $ |

#### （3）基线方法对比

| 基线类型 | 描述 |
|--------|------|
| **Mean of 8 Candidates** | 将 8 个候选模型得分取平均，衡量是否优于随机选择 |
| **Fixed@k** | 选择在整个测试集上表现最好的前 k 个模型（全局固定策略） |
| **Router@k** | 每个任务选出 top-k 候选，取其中实际表现最好的（oracle within shortlist） |

> 💡 注意：`router@k` 是诊断性指标，非实际可部署策略（除非有在线 selector）。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### （1）分类头性能（per-head accuracy）

| 任务 | 类别数 | Macro F1 |
|------|-------|---------|
| **Model Suitability** | 8 | **0.759** |
| **Task Type** | 28 | **0.837** |
| **Difficulty** | 5 | **0.789** |
| **Reasoning Mode** | ? | **0.897** |
| **Expected Output Length** | 7 | **0.788** |

> ✅ 表明 SCX Router 能准确识别任务语义特征。

#### （2）端到端路由性能（Top-1 Score）

在 **1,000-task 子集** 上比较：

| 方法 | Top-1 Score | Gain |
|------|------------|------|
| **Fixed@1**（最强固定模型） | 0.696 | — |
| **Router@1**（SCX Router） | **0.707** | **+0.011** |

> ✅ 路由器略优于最佳固定模型。

#### （3）LiveBench 子集增益（vs. 平均候选）

| 子集 | Router | Mean of 8 | Gain |
|------|--------|----------|------|
| Language | 0.779 | 0.517 | **+0.262** |
| Math | 0.738 | 0.555 | **+0.183** |
| Instruction Following | 0.863 | 0.768 | **+0.095** |
| Reasoning | 0.601 | 0.551 | **+0.050** |
| Coding | 0.500 | 0.456 | **+0.044** |
| Data Analysis | 0.540 | 0.539 | +0.001 |

> ✅ 在多数任务上显著优于平均候选。

#### （4）逐数据集对比（Top-1）

| Dataset | Fixed@1 | Router@1 | Gain |
|--------|--------|---------|------|
| LiveBench language | 0.659 | 0.726 | **+0.067** |
| LiveBench instruction following | 0.834 | 0.854 | **+0.020** |
| LiveBench coding | 0.470 | 0.500 | **+0.030** |
| LiveBench reasoning | 0.834 | 0.824 | -0.010 |
| LiveBench math | 0.738 | 0.738 | 0.000 |

> ❗ 路由并非总是胜出，在模型表现接近时优势消失。

---

### 🔍 消融实验与分析（隐含）

虽然未明确列出消融表，但从设计中可推断关键组件作用：

| 组件 | 作用 | 验证方式 |
|------|------|---------|
| **Decoder-KV Cache** | 减少重复计算，提升流式效率 | 推理延迟降低，支持 session reuse |
| **Dynamic Labels** | 支持零样本扩展新模型 | 可动态传入新模型描述而无需 retrain |
| **Separate Policy Layer** | 支持硬约束过滤与成本优化 | 实现 `U(m)` 中的安全、价格、缓存控制 |
| **Task Ontology** | 提供结构化监督信号 | 支撑 165k 合成任务生成与多信号训练 |

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **路由有效但有条件**  
   - 当候选模型在任务上有明显分歧时，SCX Router 能捕捉语义信号并做出更优选择；
   - 当所有模型表现相近时，路由增益有限甚至不如固定强模型。

2. **Decoder-KV 架构适合对话场景**  
   - 相比 encoder 或生成式 router，Decoder-KV 在多轮对话中更具效率优势，支持真正的流式推理。

3. **统一接口支持多信号预测**  
   - 同一个模型可同时输出模型适用性、任务类型、难度等，便于构建灵活的部署策略。

4. **任务本体是关键基础设施**  
   - 结构化的任务分类体系使得合成高质量、多样化、可验证的任务成为可能，推动从“prompt-centric”向“workflow-centric”评估转变。

---

### ⚠️ 局限性

1. **端到端证据有限**  
   - 当前实证仅覆盖 **direct endpoint routing**；
   - 其他模式（如 attribute-mediated、hybrid、hierarchical）仅为 proposed，尚未 end-to-end 验证。

2. **评估偏差风险**  
   - 合成任务依赖 `gpt5.6-sol` 作为 judge，可能存在 judge bias；
   - 任务生成过程可能引入 author-model 或 evaluation-shortcut bias。

3. **覆盖率不均衡**  
   - 11 个候选 endpoint 的评估任务数量不同，无法进行公平的跨模型比较；
   - 缺失观测（missing outcomes）未建模为 missing data，影响统计可靠性。

4. **未开源完整数据集**  
   - 完整的任务语料库和 outcome matrix 尚未公开，限制复现与进一步研究。

---

### 🔮 未来工作方向

1. **扩展更多模型与模态**  
   - 加入更多 LLM 家族、多模态模型、不同尺寸与价格层级。

2. **统一评估矩阵（Paired Outcome Matrix）**  
   - 所有模型在同一版本任务集上运行，形成可比的 outcome matrix。

3. **全面比较四种路由模式**  
   - 在相同 setting 下评估 direct、attribute-mediated、hybrid、hierarchical 路由的实际表现。

4. **引入在线学习与探索机制**  
   - 使用 contextual bandit、shadow evaluation 等技术缓解选择偏差，实现持续优化。

5. **发布任务本体与合成框架**  
   - 将任务生成 pipeline 开源，促进 reproducible routing benchmark 建设。

---

## 总结

SCX Router 提出了一种 **轻量、流式、零样本、多信号** 的 LLM 路由框架，通过 **Decoder-KV 分离架构** 实现高效的对话状态复用，并借助 **现实任务本体** 构建大规模合成数据。实验表明其在多个 LiveBench 子集上优于平均候选模型，并以 **0.707 vs. 0.696** 微弱优势超越最强固定模型。尽管当前证据集中在 direct routing，但它为未来构建 **智能化、成本感知、安全可控的 LLM 路由系统** 提供了坚实基础。

</details>

---

### 14. [Loom: Weaving Diagnostic Strands into Free-Text Consensus via Embedding-Space Reweighting](https://arxiv.org/abs/2609.02649)

**Authors**: Ron Begleiter, Katya Egert Berg, Gilad Saban, Gil Shabat  
**Category**: cs.AI  
**Published**: 2026-09-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.02649v1  

#### Abstract
Aggregating noisy, conflicting textual hypotheses into a reliable consensus is a fundamental challenge when deploying NLP systems in real-world industrial settings. While monolithic Large Language Model (LLM) agents offer unbounded expressivity for tasks like Root Cause Analysis (RCA), they suffer f...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Loom: Weaving Diagnostic Strands into Free-Text Consensus via Embedding-Space Reweighting**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在工业级大规模系统（如大型计算集群）中进行 **Root Cause Analysis (RCA)** 时，面临以下挑战：
- **单体 LLM Agent** 虽然表达能力强，但存在上下文限制、幻觉累积、推理延迟高、成本高昂等问题。
- **传统弱监督方法**（如 Snorkel）虽能去噪启发式规则，但仅适用于离散类别标签，无法处理自由文本形式的语义假设。

因此，如何在不依赖昂贵且不可控的迭代 LLM 推理的前提下，对开放格式、事件特定的文本假设进行数学聚合，成为一个关键难题。

---

### 🚀 提出的新方法与创新思路
论文提出 **Loom** —— 一种用于真实场景部署的生成式共识框架，其核心思想是：

#### （1）**Diagnostic Strands (DSs)**  
将诊断逻辑分解为模块化的程序化启发式单元（Python 脚本），每个 DS 输出一个**模板填充式的自由文本假设**（例如：“主机 X 出现高 CPU”），而非选择预定义类别。

#### （2）**Embedding-Space Reweighting**  
将所有触发的 DS 输出映射到连续向量空间，并通过 **迭代的嵌入中心重加权算法**（Iterative Embedding-Centroid Reweighting）动态调整各假设的重要性权重：
- 初始权重基于专家设定的可靠性元数据；
- 动态阶段计算加权质心，并根据每个输出与质心的余弦相似度更新权重；
- 最终形成有序证据列表。

#### （3）**轻量级 LLM 合成**  
仅用一次 LLM 调用，基于排序后的证据生成最终连贯的 RCA 报告，LLM 被严格约束为“合成器”，不得引入新事实。

---

### 🔍 相比现有方法的优势
| 方法 | 局限性 | Loom 的优势 |
|------|--------|-------------|
| 单体 LLM Agent（如 RCA-Agent） | 高延迟、高成本、易产生幻觉、推理路径不可审计 | **低延迟（~26× 加速）、低成本、确定性输出、可审计性强** |
| 弱监督（Snorkel 等） | 仅支持离散分类，无法处理自由文本 | 支持**开放形式、带实体/时间/指标的自由文本聚合** |
| 多智能体辩论（Multi-Agent Debate） | 多轮交互导致 token 成本爆炸 | 冲突解决在**数学空间完成**，无需多轮 LLM 交互 |

> ✅ **核心突破**：首次将弱监督范式扩展至**结构化自由文本假设**的聚合，同时保留 LLM 的自然语言生成能力。

---

## 2. 核心实验方法和设置

### 📚 数据集
使用公开基准 **OpenRCA**（Xu et al., 2025），包含四个子数据集：
- **Bank**（136 个事件）
- **Telecom**（51 个事件）
- **Market-1**（70 个事件）
- **Market-2**（78 个事件）

这些数据集模拟真实工业环境中的故障场景，涵盖日志、遥测等多源异构数据。

---

### ⚙️ 实验设置与评估指标

#### 评估方式
- **Strict Accuracy**：完全匹配根因组件、原因和时间戳。
- **Partial Accuracy**：部分正确（如组件或原因匹配）。
- 所有实验在 `temperature=0` 下运行，确保结果可复现。

#### 基线方法
- **RCA-Agent + Claude 4.6**：基于 ReAct 的自主代理，平均每次事件调用约 62 次 LLM。
- **Oracle Score**：理想情况下的上限——若 Top-K 候选中包含真实根因，则视为可通过人工干预识别。

#### Loom 设置
- 使用 **Claude 4.6** 或 **Llama-3.1-8B** 作为合成 LLM。
- 所有 DS 在离线阶段由领域专家编写或通过 LLM 从历史工单自动提取。
- 聚合过程独立于 LLM，仅最后一步使用 LLM 进行文本整合。

---

## 3. 主要实验结果和性能指标

### 📊 性能对比（来自 Table 1 和 Table 2）

| 数据集 | RCA-Agent (Strict) | Loom + Claude 4.6 (Strict) | 速度提升 |
|-------|---------------------|----------------------------|----------|
| Bank | 40.44% | **38.97%** | ~26× |
| Market-2 | 35.90% | **35.90%** | ~26× |
| Market-1 | 40.00% | 28.57% | ~26× |
| Telecom | 41.18% | 29.41% | ~26× |

> 💡 **关键发现**：
> - 在 **Bank 和 Market-2** 上，Loom **达到甚至略微超过** RCA-Agent 的部分准确率（Partial Acc.: 51.22% vs 49.15%）；
> - 在所有数据集上，Loom **仅需 1 次 LLM 调用**，而 RCA-Agent 平均需要 **62 次**，耗时从近 10 分钟降至 **~22 秒**；
> - 若使用 **Llama-3.1-8B** 小模型作为合成器，速度进一步提升至 **~33×**。

---

### 🔍 消融实验（Ablation Study on Bank Dataset）

| 配置 | Strict Acc. | Partial Acc. |
|------|-------------|--------------|
| 完整 Loom 系统 | **38.97%** | **51.22%** |
| 移除迭代重加权（w/o Iter. Reweighting） | 33.09% | 42.70% |
| 移除静态冗余检测（w/o Redundancy Det.） | 44.85% | 53.36% |
| 两者都移除（Raw LLM Synthesis） | 35.29% | 44.49% |

> ✅ **结论**：
> - **迭代重加权算法贡献显著**：带来 +5.88 pp 的严格准确率提升；
> - **静态冗余检测反而有害**：在专家维护的小规模 DS catalog 中，docstring 相似性判断过于粗糙，会误删有效假设。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Loom 占据精度-效率帕累托前沿（Pareto Frontier）**
   - 在多数场景下实现与最先进 Agent 相当的准确性，同时实现 **26–33× 的推理加速**；
   - 特别适合对延迟敏感、资源受限的工业部署环境。

2. **共识机制解耦于 LLM 规模**
   - 使用 **8B 参数的小型本地模型**即可取得接近 Claude 4.6 的性能；
   - 表明冲突解决主要由数学聚合完成，LLM 仅承担摘要任务。

3. **确定性共识增强 SME 信任**
   - 输出可审计、可追溯，便于运维人员理解和验证；
   - 对比黑箱式 Agent 更易被领域专家采纳。

4. **Oracle 分析揭示瓶颈所在**
   - 在 Market-1 和 Telecom 上，虽然 Loom 的最终准确率较低，但 Oracle 显示正确根因常出现在候选前列（Market-1 达 70%）；
   - 说明问题不在聚合阶段，而在 **single-shot synthesis 的判别能力不足**。

---

### ⚠️ 方法的局限性

| 局限性 | 说明 |
|--------|------|
| **准确性差距（Market-1 & Telecom）** | 在复杂噪声密集场景下，single-shot LLM 难以区分高度相关的故障类型（如“同一家族原因混淆”）。 |
| **语义覆盖有限** | 依赖预定义的 DS catalog，无法像通用 Agent 那样零样本应对全新问题。 |
| **冷启动挑战** | 新型“黑天鹅”故障需人工或离线流程添加新 DS，响应较慢。 |
| **Judge Overfocus 现象** | 当多个节点出现强异常信号时，LLM 可能过度聚焦于最突出但非根本的原因。 |
| **嵌入质量依赖性强** | 若 embedding model 无法捕捉技术细节间的细微差异，可能导致错误聚类。 |

---

### 🔮 未来工作方向

1. **混合架构设计**
   - 使用 Loom 快速筛选 Top-K 候选根因；
   - 再由轻量级 Agent 进行最终细粒度辨析，兼顾效率与精度。

2. **动态冗余检测**
   - 改进冗余判断机制：从静态 docstring 匹配转向**基于输出内容的动态语义比较**。

3. **增强小型合成器能力**
   - 设计更有效的提示工程或微调策略，提升小模型在 disambiguation 上的表现。

4. **自动化 DS 构建闭环**
   - 建立从失败案例 → 自动归纳新 DS → 测试上线的完整 pipeline，缓解冷启动问题。

5. **跨域迁移潜力探索**
   - 将 Loom 框架推广至医疗诊断、金融风控等其他需要融合专家知识与自然语言推理的领域。

---

## ✅ 总结

**Loom** 是一项面向工业落地的创新性工作，成功地在 **表达力、效率、可信度**之间取得了平衡。它不是追求“全能 AGI 式诊断”，而是构建了一个**可解释、高效、可控**的诊断系统，在真实环境中更具实用价值。其核心理念——**将冲突解决前移到数学空间，让 LLM 专注合成而非推理**——为未来 LLM 系统设计提供了重要启示。

</details>

---

### 15. [Grounded, Compute-Efficient LLM Policy Agents for Energy-Poverty Equity in Physically-Constrained Peer-to-Peer Energy Markets](https://arxiv.org/abs/2609.01918)

**Authors**: Kunal Jadhav, Siddhesh More  
**Category**: cs.CL  
**Published**: 2026-09-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.01918v1  

#### Abstract
Energy poverty is nearly absent from NLP-for-social-good, and the little existing work is either static retrieval/QA or relies on carbon-intensive cloud LLMs, a self-defeating "computational irony" for a humanitarian setting. We present EqGrid, a closed-loop simulation in which a low-frequency, open...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Grounded, Compute-Efficient LLM Policy Agents for Energy-Poverty Equity in Physically-Constrained Peer-to-Peer Energy Markets*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文聚焦于**能源贫困（energy poverty）** 这一在 NLP-for-Social-Good 领域长期被忽视的社会议题。现有研究存在两大缺陷：
- 多为静态检索或问答系统（如 RAG QA），无法进行动态干预；
- 依赖大型云上 LLM，其高碳足迹与“可持续能源”目标形成“计算讽刺”（computational irony）。

此外，现有的 P2P 能源市场研究通常忽略以下要素的联合建模：
- 社会经济层面的真实家庭画像（personas）
- 物理电网约束（如电压、线路容量）
- 公平性（尤其是 energy burden 不平等）
- AI 自身的推理能耗与碳成本

### 提出的新方法与思路
作者提出 **EqGrid** ——一个闭环模拟框架，首次将以下四个模块集成在一个系统中：

| 模块 | 功能 |
|------|------|
| **Empirically Grounded Personas** | 基于 EU-SILC 数据构建具有真实社会经济属性的家庭画像（收入、住房效率、取暖方式等），并通过 LLM 推导活动模式生成负荷曲线 |
| **LLM Policy Agent (低频策略层)** | 使用开放权重的 LLM 每 6 小时设定一次价格上下限、碳配额和定向补贴，作为市场调控政策 |
| **MARL Traders (高频交易层)** | 基于 MAPPO 的多智能体强化学习代理，在 LLM 设定的边界内执行连续双向拍卖（continuous double auction） |
| **Physical Grid Layer (物理安全层)** | 在 IEEE-33 总线系统上实现带 Dynamic Operating Envelopes (DOE) 的潮流校验，确保所有操作满足电压和热极限 |

#### 创新设计亮点：
- **Decoupled-Safety Design**：LLM 只输出经济边界，不直接控制物理动作；由确定性的 validate-and-project 网关强制投影到可行集，实现零电网违规。
- **Compute-Efficiency Frontier**：首次量化衡量从 235B 教师模型压缩至 <1B 模型对社会公平性能的影响，并结合每决策的 energy/carbon 成本进行权衡分析。
- **Social Impact Measurement Suite**：引入正式的能源贫困公平指标（Energy Burden, Gini of EB, LIHC）并定义 **Carbon-adjusted SROI**，显式扣除 AI 自身碳成本。

### 相比现有方法的优势
| 维度 | EqGrid 的优势 |
|------|----------------|
| **社会影响测量** | 同时评估干预效果与 AI 自身碳代价，避免“以环境换公平”的悖论 |
| **部署可行性** | 支持 sub-1B 模型在笔记本运行，适合边缘部署，降低对云服务依赖 |
| **安全性保障** | 通过解耦设计实现零 grid violation，而直接 LLM 控制下出现 55 次违规 |
| **公平性针对性** | 明确优化 energy burden inequality，而非通用 fairness score |

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据来源 | 用途 |
|--------|------|
| **EU-SILC Marginals (Hungary)** | 构建家庭 persona 的统计分布（等效收入、家庭规模、住宅能效等级、取暖类型、是否拖欠账单、无法保暖） |
| **UCI ML Repository (Individual household electric power consumption, 2012)** | 用于验证生成负荷曲线的**日变化形状和用电水平合理性**（尽管地域与时段不同，仅作粗略 sanity check） |
| **Winter PV Generation Profile** | 为 prosumer 家庭添加光伏发电曲线 |

### 实验设置
- **仿真场景**：布达佩斯居民社区，连接 IEEE-33-bus radial feeder，每个 bus 表示一个低压负载区
- **时间尺度**：
  - 高频交易：每小时一轮连续双向拍卖
  - 低频政策更新：每 6 小时由 LLM 更新一次政策参数
- **模型频率解耦**：LLM 作为 ex ante 政策制定者，不参与实时交易，使其可小型化
- **训练协议**：
  - MARL 采用 centralised training with decentralised execution (CTDE)，使用 shared-parameter MAPPO
  - LLM 通过 prompt 工程指导生成 JSON 格式的政策建议（price ceiling/floor, subsidy weight, carbon allowance）

### 评估指标
| 指标 | 定义 | 目标 |
|------|-----|------|
| **Energy Burden (EB)** | $ \text{EB}_i = \frac{\text{net annual energy cost}}{\text{income}_i} $ | 越低越好 |
| **Gini of EB** | 衡量社区内 energy burden 分配不均程度 | 越低表示越公平 |
| **LIHC Prevalence** | 收入低于中位数且 EB > 10% 的家庭数量 | 越少越好 |
| **Daily Bill Cost (€/day)** | 平均每日净支出 | 越低越好 |
| **CaO₂-adjusted SROI** | $ \text{SROI}_{\text{net}} = (\Delta V_{\text{social}} - C_{\text{carbon}})/I_{\text{initial}} $，其中 $ C_{\text{carbon}} $ 是 LLM 推理的碳成本 | 越高越好 |
| **Energy/CO₂ per Decision** | 每次 LLM 决策消耗的能量（Wh）与碳排放（kgCO₂e） | 越低越可持续 |
| **Grid Violations** | 电压越限或线路过载次数 | 应为 0 |

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **NONE** | 无任何政策干预，仅 MARL 双向拍卖市场 |
| **RULE** | 手工设计规则：当不平等上升时收紧价格上限，按欠费率调整补贴 |
| **LLM (teacher)** | 235B 参数教师模型（qwen3-235b-a22b）作为最强基准 |
| **LLM (compressed)** | 多种小型模型（down to 0.8B），测试性能保留情况 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 1）

| Policy | Gini(EB) ↓ | Mean EB ↓ | LIHC ↓ | Daily Bill (€/day) ↓ |
|--------|------------|-----------|--------|-----------------------|
| none   | 0.351 ± 0.017 | 0.247 ± 0.043 | 14.8 | 148.0 ± 12.6 |
| rule   | 0.330 ± 0.012 | 0.217 ± 0.031 | 14.2 | 135.7 ± 9.4 |
| **llm** | **0.305 ± 0.023** | **0.177 ± 0.030** | **12.4** | **120.3 ± 10.8** |

> ✅ LLM 政策显著优于 no-policy 和 rule baseline，在所有公平性和成本指标上均取得最佳表现。

### 与基线方法的对比结果
- **vs. NONE**：
  - Gini(EB) 下降 **0.046**（p=0.009）
  - 日均账单减少 **€27.7**（p<0.001）
  - SROI 提升至 **101.1**（vs. ~0）
- **vs. RULE**：
  - 成本再降 **€15.4**（p=0.001）
  - SROI 提高 **+56**（p=0.001）
  - Gini 进一步下降 0.025（方向正确但未达统计显著性）

👉 表明 LLM 的 context-sensitive reasoning 提供了超越透明规则的额外价值。

### Compute-Efficiency Frontier（Table 2 & Figure 2）

| Model | Act. Params | Gini(EB) | Equity Retained (%) | Wh/Decision | Energy Reduction vs Teacher |
|-------|-------------|----------|----------------------|--------------|----------------------------|
| qwen3-235b (teacher) | 220B | 0.302 | 100% | 0.2745 | ×1 |
| qwen3-30b-a3b (MoE) | 3B | 0.305 | 95% | 0.0292 | **~9×** |
| olmo3-7b | 7B | 0.306 | 92% | 0.0784 | ~3.5× |
| **qwen35-0p8b** | **0.8B** | **0.306** | **92%** | **0.0113** | **~24×** |
| gemma4-e2b | 2B | 0.310 | 85% | 0.0262 | ~10× |
| qwen36-27b (dense) | 27B | 0.336 | 31% | 1.1071 | **×4 higher than teacher!** |

> 🔍 发现：**最小的 0.8B 模型仍保留 92% 的公平增益，能耗仅为教师模型的 1/24**  
> ❌ 反例：较大的 dense reasoning model（27B）因输出截断导致失败，实际可用决策极少

### 消融实验结果

#### （1）Persona Grounding Ablation
- 对比 **GROUNDED** vs **SYNTHETIC** personas
- 结果：补贴-负担对齐度 alignment 为 0.79 vs 0.70，差异不显著（p=0.06）
- 结论：经验接地提升了构造有效性（construct validity）和现实感，但不是性能跃迁的关键驱动因素

#### （2）Safety Ablation (H4)
- **With DOE Gate**：0 次电网违规
- **Without (LLM-direct-control)**：55 次违规
- 👉 验证了解耦安全机制的必要性

#### （3）Subsidy Mechanism Analysis
- 回归显示：Gini(EB) 与政策中的 subsidy weight 强相关（R²=0.985）
- 当前实验本质是测试模型能否输出合理缩放的 scalar，而非完整财政机制的有效性

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **LLM 可有效用于能源贫困干预**：相比无干预和手工规则，LLM 政策显著降低 energy burden inequality（Gini↓28%）、平均负担（↓28%）、LIHC 数量（↓16%）和总成本。
2. ✅ **小型化模型极具潜力**：**0.8B 模型即可保留 92% 的公平收益，能耗仅为大模型的 ~1/24**，支持本地化、低碳部署。
3. ✅ **推理密集型模型不适合此类任务**：即使更大参数的 dense reasoning model 也会因输出截断而失效，且单位能耗极高，**在效率前沿上被完全支配（Pareto-dominated）**。
4. ✅ **解耦安全机制至关重要**：LLM 仅设边界，由 deterministic solver 执行，可实现 **zero grid violation**，解决 LLM hallucination 的安全隐患。
5. ✅ **必须核算 AI 自身碳成本**：提出 CaO₂-adjusted SROI，使社会回报评估更全面、负责任。

### 局限性（Limitations）
1. **Persona 接地不够强**：基于公开边际分布而非微观数据（microdata），缺乏个体级匹配。
2. **负荷验证参考数据跨区域/时期**：使用法国单一住户数据验证匈牙利群体负荷，仅作粗略检查。
3. **仿真规模有限**：单个馈线、单一冬季日，缺乏季节性与大规模网络扩展。
4. **能耗为估算值**：未实测硬件功耗，依赖 active-parameter × token-count 的代理模型。
5. **补贴为无资金转移**：未建模税收或预算平衡，影响政策净福利判断。
6. **纯仿真研究**：尚未实地部署，需与公用事业公司及受影响社区合作验证。

### 未来工作方向
1. 使用 **Low Carbon London** 等多住户、同区域负荷数据集进行更精确的分布匹配验证。
2. 引入 **revenue-neutral subsidy mechanism**，重新评估财政可持续性下的公平-成本权衡。
3. 开展 **on-device power profiling**（如 Apple Silicon 能耗计数器或 CodeCarbon），获取真实 energy/carbon 数据。
4. 探索 **federated learning + local LLM** 架构，进一步增强隐私保护与去中心化能力。
5. 推进 **field trials with utilities**，实现从仿真到现实世界的过渡。

---

> 📢 **最终主张**：对于低频、高影响力的政策决策任务（如能源补贴调控），**small, open, on-device LLMs 是比 large cloud models 更优选择**——它们不仅足够聪明，而且足够绿色、足够安全。EqGrid 提供了一个可复制、可审计、兼顾社会公平与 AI 可持续性的新范式。

</details>

---

### 16. [Federated Learning on the American Science Cloud using APPFL](https://arxiv.org/abs/2609.02238)

**Authors**: Zilinghan Li, Abhijit Chunduru, Harinarayan Krishnan, Eric Chagnon, Peter Nugent, Kibaek Kim, Ravi Madduri  
**Category**: cs.DC  
**Published**: 2026-09-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.02238v1  

#### Abstract
The American Science Cloud (AmSC), established under the Genesis Mission of the U.S. Department of Energy (DOE), aims to integrate DOE high-performance computing systems, experimental facilities, and data resources into a single, coordinated, AI-driven discovery platform. AmSC's early services focus...

---

### 17. [PEARL: Path-Entity Aligned Relational Learning with Contextual Subgraphs for Inductive Knowledge Graph Completion](https://arxiv.org/abs/2609.02216)

**Authors**: Yunchi Yang, Longlong Li, Cunquan Qu  
**Category**: cs.AI  
**Published**: 2026-09-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.02216v1  

#### Abstract
Inductive knowledge graph completion (IKGC) aims to predict missing links involving entities unseen during training, requiring models to learn transferable relational and structural patterns. Existing subgraph- and path-based approaches often encode relational paths independently of their surroundin...

---

### 18. [PGPO: Potential-Guided Policy Optimization for Multi-Turn Agentic Tasks](https://arxiv.org/abs/2609.02236)

**Authors**: Yuyao Zheng, Haipeng Sun, Junwei Bao, Lemao Liu, Hongfei Jiang, Yang Song, Dejing Dou  
**Category**: cs.AI  
**Published**: 2026-09-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.02236v1  

#### Abstract
Group-based reinforcement learning (RL) has become an effective paradigm for LLM post-training, but in multi-turn agentic tasks with sparse terminal rewards, it often provides coarse credit for intermediate actions. To obtain more fine-grained credit assignment, recent work such as GiGPO introduces ...

---

### 19. [CoMerge: Conflict-Driven Preference Optimization for Multi-Task Model Merging](https://arxiv.org/abs/2609.02273)

**Authors**: Mingjie Zheng, Zihao Chen, Wenqing Chen, Weile Yuan, Zhixuan Chu, Jianxing Yu, Zibin Zheng  
**Category**: cs.AI  
**Published**: 2026-09-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.02273v1  

#### Abstract
Model merging provides an efficient paradigm for constructing multi-task large language models (LLMs) without full model retraining, yet it remains challenged by parameter interference. While existing methods aim to preserve the capabilities of individual expert models and mitigate interference, the...

---

### 20. [Act More, Decide Less: Skill-Guided Adaptive Action Chunking for Long-Horizon LLM Agents](https://arxiv.org/abs/2609.02042)

**Authors**: Yanting Yang, Can Jin, Jinman Zhao, Jiahao Wu, Yang Zhou, Zhepeng Wang, Zhendong Wang, Mu Zhou, Dimitris N. Metaxas  
**Category**: cs.LG  
**Published**: 2026-09-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.02042v1  

#### Abstract
Large language model (LLM) agents for long-horizon interactive tasks typically follow a ReAct-style protocol, issuing one primitive action per LLM round. While this enables frequent replanning, it is inefficient for long-horizon tasks where many rounds are spent on routine action sequences. A natura...

---

### 21. [DynG-Diff: A State-Aware Dynamic Guidance Diffusion Framework for Probabilistic Time Series Forecasting](https://arxiv.org/abs/2609.02068)

**Authors**: Zhente Zhang, Zhengwei Ni, Wei Fan  
**Category**: cs.LG  
**Published**: 2026-09-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.02068v1  

#### Abstract
Probabilistic multivariate time series (MTS) forecasting is crucial for modeling complex dynamical systems. However, existing diffusion-based methods rely on task-specific conditional paradigms that lack flexibility and struggle with inherent "information heterogeneity"--the significantly varying no...

---

### 22. [Induction and Inquiry via Probabilistic Reasoning over Language and Code](https://arxiv.org/abs/2609.01815)

**Authors**: Wasu Top Piriyakulkij, Sam Acquaviva, Cassidy Langenfeld, Joshua Tenenbaum, Kevin Ellis  
**Category**: cs.AI  
**Published**: 2026-09-03  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.01815v1  

#### Abstract
How humans grow and maintain abstract knowledge from the sparse, streaming noisy data of experience is a longstanding challenge in cognitive science. Any computational account must satisfy at least three desiderata: It must be (1) data-efficient and compute-efficient, (2) capture gradations of uncer...

---

### 23. [Tri-Band Channel Measurement-Enabled Multi-Layer Digital Twin for Terahertz Wireless Data Centers](https://arxiv.org/abs/2609.01699)

**Authors**: Mingjie Zhu, Ziming Yu, Guangjian Wang, Chong Han  
**Category**: cs.LG  
**Published**: 2026-09-03  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.01699v1  

#### Abstract
The rapid growth of AI computing has driven increasing demands for flexible and high-capacity data-center interconnections. Owing to its ultra-wide bandwidth and high spatial reuse capability, terahertz (THz) communication has emerged as a promising solution for future wireless data centers, while d...

---

### 24. [CAT-Flow: Curvature-Adaptive sTeps for Flow Matching](https://arxiv.org/abs/2609.01746)

**Authors**: Qinchan Li, Pedro Cisneros-Velarde, Keru Fu, Samuel Antunes Miranda, Sharan Vaswani, Hao Zhang  
**Category**: cs.LG  
**Published**: 2026-09-03  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.01746v1  

#### Abstract
Flow Matching has emerged as a leading framework for generative modeling, powering state-of-the-art systems such as FLUX and Stable Diffusion 3.5. However, the iterative nature of its ODE-based sampling process creates a fundamental efficiency bottleneck: the quality of generated samples is highly s...

---

### 25. [Measurement-Driven Sub-Network Selection for On-Premise Retrieval-Augmented Factory Agents](https://arxiv.org/abs/2609.02760)

**Authors**: Vasileios Rizeakos, Georgios Paisios, Alexandros Machairas, Michael Birbas, Athanasios Bachoumis  
**Category**: cs.AI  
**Published**: 2026-09-03  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.02760v1  

#### Abstract
On-premise assistants can give factory workers conversational access to machine documentation, but models capable of the task rarely fit shop-floor hardware. We show that after structural compression and retrieval-grounded adaptation, model size is no longer a reliable predictor of adapted answer qu...

---

### 26. [HyperStyler: Low-resource Authorship Style Transfer via Context-aware Style Navigation and Hypernetworks](https://arxiv.org/abs/2609.02772)

**Authors**: Jongkyung Shin, Minguk Jeon, Chanwoo Park, Chiehyeon Lim  
**Category**: cs.CL  
**Published**: 2026-09-03  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.02772v1  

#### Abstract
Low-resource authorship style transfer (LAST) aims to rewrite text into the style of an arbitrary target author using only a few reference examples while preserving the original meaning. Existing methods often struggle to achieve both high style fidelity and semantic preservation because they compre...

---

### 27. [EarlyEval: Cheaper Agent Evaluation via Early Outcome Prediction](https://arxiv.org/abs/2609.02783)

**Authors**: Yuling Shi, Zhensu Sun, Junsen Dong, Chengcheng Wan, David Lo, Xiaodong Gu  
**Category**: cs.CL  
**Published**: 2026-09-03  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.02783v1  

#### Abstract
Evaluating LLM agents is essential for guiding their development, yet it has grown prohibitively expensive: a single pass of a frontier model over an agentic benchmark can cost hundreds to thousands of dollars, a price paid repeatedly across iterative development cycles. Prior efforts, centered on b...

---

### 28. [CREDIT: Cost-guided Reduction-reuse with Efficient DSMEM Inter-CTA Tiling](https://arxiv.org/abs/2609.01864)

**Authors**: Zhengxiong Li, Tsung-Wei Huang, Umit Ogras  
**Category**: cs.DC  
**Published**: 2026-09-03  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.01864v1  

#### Abstract
NVIDIA distributed shared memory (DSMEM) enables direct shared-memory access within a thread block cluster. However, cluster synchronization, remote access, and resource costs make it difficult to determine when DSMEM improves performance. To fill this gap, we propose CREDIT, a cost-guided framework...

---

### 29. [Compositional Spectral Prompts for LLM-based Online Time Series Forecasting](https://arxiv.org/abs/2609.02093)

**Authors**: Seungyoon Choi, Hyunchul Kim, Jae-Gil Lee, Chanyoung Park  
**Category**: cs.LG  
**Published**: 2026-09-03  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.02093v1  

#### Abstract
To address the sequential and evolving nature of time series, the Online Time Series Forecasting (OTSF) task has been extensively studied in multiple domains. Existing research focuses on adapting to non-stationary environments by employing memory buffer-based retrieval strategies. However, we obser...

---

### 30. [A Computational Comparison of Fourier Spectral Differentiation and Spatial Automatic Differentiation in Periodic Physics-Informed Neural Networks](https://arxiv.org/abs/2609.02110)

**Authors**: Xilai Liang, Zhao Zhang  
**Category**: cs.LG  
**Published**: 2026-09-03  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.02110v1  

#### Abstract
Physics-informed neural networks (PINNs) commonly evaluate the spatial derivatives appearing in partial differential equation residuals using automatic differentiation (AD), whose computational and memory costs can become substantial when multiple or high-order derivatives are required. We perform a...

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

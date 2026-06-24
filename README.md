# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-06-24 08:46:58 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [CompressKV: Semantic-Retrieval-Guided KV-Cache Compression for Resource-Efficient Long-Context LLM Inference](https://arxiv.org/abs/2606.24467)

**Authors**: Xiaolin Lin, Jingcun Wang, Olga Kondrateva, Yiyu Shi, Bing Li, Grace Li Zhang  
**Category**: cs.AI  
**Published**: 2026-06-24  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.24467v1  

#### Abstract
Long-context large language model (LLM) inference is increasingly constrained by the memory footprint and decoding cost of key-value (KV) caches, limiting sustainable deployment on resource-constrained hardware. Existing KV cache eviction methods typically apply heuristic token scoring over all head...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：CompressKV: Semantic-Retrieval-Guided KV-Cache Compression for Resource-Efficient Long-Context LLM Inference**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在基于 **Grouped Query Attention (GQA)** 的大语言模型（LLM）中，长上下文推理面临 **Key-Value (KV) cache** 内存占用高和解码延迟大的挑战。现有的 KV cache eviction 方法通常对所有注意力头进行统一评分（如求和或平均），忽略了不同注意力头的功能异质性（head heterogeneity）。这导致“流式头”（Streaming Heads）主导缓存保留决策，仅保留序列首尾 token，而中间重要的语义信息被错误地丢弃，从而损害模型性能。

### **提出的新方法与新思路**
作者提出了 **CompressKV**，一种面向 GQA 架构的资源高效 KV cache 压缩框架，其核心创新包括：

- **Semantic Retrieval Head (SRH) 识别机制**  
  提出并识别一类特殊的注意力头——**Semantic Retrieval Heads (SRHs)**，它们不仅能捕捉答案 token 的直接复制行为（copy-and-paste），还能关注整个答案跨度及其语义邻域（如“eat a ___ sandwich”中的上下文线索）。通过聚合整个答案 span 上的注意力质量来打分，而非依赖传统的 top-1 或 top-k 准则。

- **SRH 驱动的 token 选择策略**  
  利用每层中识别出的 top-k SRHs 来指导重要 token 的保留，所有头共享由 SRHs 决定的关键 token 集合，避免 Streaming Heads 主导压缩过程。

- **误差感知的层自适应缓存分配（Error-Aware Layer-Adaptive Allocation）**  
  在离线阶段计算每一层在 KV 压缩下的输出重建误差（使用 Frobenius 范数衡量 full-cache 与 compressed-cache 输出差异），据此动态分配各层的 KV 缓存预算，优先保障误差高的关键层。

### **相比现有方法的优势**
- ✅ 更精准地保留语义关键 token，尤其适用于中长距离依赖任务；
- ✅ 离线完成 SRH 识别与层预算分配，**不引入在线计算开销**；
- ✅ 显著优于主流 KV eviction 方法（如 SnapKV、CAKE、HeadKV 等）在极端压缩场景下的表现；
- ✅ 可与其他优化技术（如量化、prefilling 加速）正交结合，进一步提升效率。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **LongBench**：包含 16 个长上下文子任务，涵盖单文档 QA、多文档 QA、摘要、少样本学习、代码补全等，用于综合评估模型能力。
- **Needle-in-a-Haystack (NIAH)**：将目标答案“needle”嵌入大量无关文本“haystack”中，测试模型从超长上下文中检索关键信息的能力。

此外还进行了基于 **masking-based ablation** 的因果分析以验证 SRH 的作用。

### **实验设置与评估指标**
- **模型**：Llama-3.1-8B-Instruct、Mistral-7B-Instruct-v0.3、Qwen2.5-14B-Instruct 和 Qwen2.5-32B-Instruct。
- **KV cache 预算范围**：每层平均保留 128–2048 个 token（Bper-layer），极端情况下低至 **0.7% 的原始 KV 存储**。
- **评估指标**：
  - LongBench：平均得分（Acc. %）
  - NIAH：准确率（Accuracy）
  - 推理效率：端到端延迟、首次 token 时间、峰值 GPU 内存、吞吐量

### **基线方法对比**
共比较六种主流 KV cache eviction 方法：
- **StreamingLLM**：仅保留首尾 token
- **SnapKV**：基于滑动窗口内注意力得分聚类
- **PyramidKV**：金字塔式信息漏斗压缩
- **CAKE**：基于注意力方差与级联机制
- **HeadKV / AdaKV**：支持 head-level 或 group-level 缓存分配

所有方法均在相同 prefilled 设置下运行（window_size=8, kernel_size=5），确保公平性。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 场景 | 方法 | KV Cache 占比 | 性能保留率 |
|------|------|----------------|-------------|
| **LongBench (QA)** | CompressKV | **3%** | > **97% 全缓存性能** |
| **NIAH** | CompressKV | **0.7%** | 达到 **90% 基线准确率** |
| **LongBench (总体)** | CompressKV | 19% | 维持 **>99% 全缓存性能** |

> 注：这些结果是在 Llama 和 Mistral 系列模型上取得的。

### **与基线方法的对比结果**
#### ✅ **LongBench 平均得分（Table 1）**
在 **256-token per layer** 极限压缩下：
- CompressKV 在所有模型上均达到最高分：
  - Llama-3.1-8B: **46.71** vs. 第二名 CAKE (46.30)
  - Mistral-7B: **45.43** vs. CAKE (44.73)
  - Qwen2.5-32B: **44.73** vs. CAKE (44.49)

> 图 4 显示，在小缓存规模（<512）时 CompressKV 提升最为显著。

#### ✅ **NIAH 检索任务（Figure 5）**
- 在 Llama-3.1-8B 上，当 KV budget = 256（仅占全缓存 **0.7%**）时：
  - CompressKV 仍保持 **~90% 准确率**
  - 其他方法（如 AdaKV、HeadKV）在此预算下严重下降

### **消融实验结果（Ablation Study）**
#### （1）组件有效性（Table 3a）
在 Mistral-7B 上固定 256-token 预算：
| 方法 | 准确率 (%) |
|------|------------|
| SnapKV（基线） | 43.76 |
| + SRH Selection | 44.96 (**+1.2**) |
| + SRH + Layer Alloc. | **45.43 (+1.67)** |

👉 表明两个模块互补且有效。

#### （2）每层 SRH 数量影响（Table 3b）
- 最优为 **top-4 SRHs per layer**
- 多于 4 个后性能饱和甚至轻微下降（如 top-24 下降 0.66 pts）

#### （3）SRH vs. TRH 对比（Table 2）**
- 使用传统 Retrieval Head (TRH) 时 LongBench 得分为 44.72
- 使用 SRH 后提升至 **44.96**（+0.24）
- 屏蔽 top SRH 导致 NIAH 性能大幅下降，说明其对事实检索至关重要

---

## **4. 关键结论和发现**

### **主要发现**
1. **注意力头功能异质性不可忽视**：Streaming Heads 主导会导致关键中段 token 被误删。
2. **Semantic Retrieval Heads 是更鲁棒的 token 重要性探测器**：通过 span-level 注意力聚合，能捕获深层语义关联，优于 peak-driven 的 TRH。
3. **离线误差估计可用于高效层间资源分配**：无需在线 profiling，即可实现高性能的 adaptive budget 分配。
4. **CompressKV 实现了极高压缩比下的性能稳定**：在仅保留 **0.7%-3% KV cache** 的情况下仍接近全缓存性能。

### **方法的局限性**
- SRH 的识别依赖一个校准数据集（calibration dataset），虽然只需一次离线执行，但仍需额外标注或构造数据；
- 当前设计针对 GQA 架构优化，是否完全适配 Multi-Head Attention (MHA) 尚未验证；
- 极端压缩下仍可能出现罕见的幻觉或遗漏，尤其是在多 needle 场景中。

### **未来工作方向**
- 探索自动化的 SRH 动态更新机制，适应不同输入分布；
- 扩展至多模态 LLM 中的 cross-modal KV cache 压缩；
- 结合稀疏激活与 KV 压缩，构建端到端的绿色推理系统；
- 研究如何将 SRH 发现应用于模型微调或提示工程中。

---

> 🔗 **开源地址**：https://github.com/TUDa-HWAI/CompressKV  
> 📄 **论文链接**：arXiv:2606.24467

</details>

---

### 2. [Accelerating Disaggregated RL for Visual Generative LLMs with Diffusion-Based Parallelism and Trainer-Assisted Generation](https://arxiv.org/abs/2606.24369)

**Authors**: Sijie Wang, Zhengyu Qing, Zhiqiang Tan, Yiming Yin, Yeqing Zhang, Yaoyuan Wang, Qiang Wang, Xiaowen Chu, Shaohuai Shi  
**Category**: cs.AI  
**Published**: 2026-06-24  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.24369v1  

#### Abstract
Reinforcement learning (RL) has become a dominant post-training paradigm, driving the emergence of high-performance RL systems such as veRL for autoregressive large language models (LLMs). In parallel, diffusion-oriented RL algorithms, e.g., DanceGRPO and FlowGRPO, have rapidly expanded the scope of...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Accelerating Disaggregated RL for Visual Generative LLMs with Diffusion-Based Parallelism and Trainer-Assisted Generation

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前用于视觉生成大模型（如扩散模型）的 **Reinforcement Learning (RL)** 后训练系统（如 veRL-Omni、GenRL）大多采用**集中式架构（colocated architecture）**，即生成（rollout）和训练（training）共享同一组 GPU 资源。这种设计虽然简化同步，但存在以下问题：

- **资源耦合严重**：rollout 和 trainer 无法独立扩展，难以适应异构硬件环境；
- **利用率低**：由于生成与训练阶段交替执行，导致大量 pipeline bubbles（空闲时间），GPU 利用率不足；
- **调度不灵活**：缺乏对细粒度任务并行的支持，限制了吞吐量提升。

尤其在 **Diffusion-RL** 场景中（如 DanceGRPO、FlowGRPO），生成过程是时序性的（denoising steps 逐步去噪），而训练可并行处理多个 timestep，二者计算负载不对称，加剧了瓶颈。

---

### 提出了什么新方法或新思路
本文提出 **DigenRL** —— 一种面向扩散式生成 LLM 的**解耦式（disaggregated）RL 框架**，其核心创新包括：

#### （1）**Generation-Axis Pipeline (GAP) + Timestep Parallelism (TSP)**
- **GAP**：在生成端沿“采样维度”而非“批次维度”划分微批次（micro-batch），实现更细粒度的流水线并行，增加 pipeline overlap。
- **TSP**：在训练端将多个 denoising timesteps 打包成一个 timestep batch，并行计算损失，复用参数通信开销（如 FSDP all-gather），显著提升 trainer 的 GPU 利用率。

> ✅ 优势：打破传统仅依赖 batch dimension 并行的局限，在小批量 Diffusion-RL 中仍能获得高并发。

#### （2）**Elastic Trainer-Assisted Generation (TAG)**
- 允许 trainer 在空闲时动态参与部分 rollout 生成任务（通过上下文切换加载 generator 模型）；
- 弹性利用 trainer 的 bubble 时间，缓解 generator 瓶颈。

> ✅ 优势：避免资源闲置，特别适用于 trainer 快于 generator 的场景。

#### （3）**Trajectory-Consistent Stale Synchronization (TCSS)**
- 放宽严格同步约束，允许 generator 使用最多一个旧策略版本（stale policy）提前生成下一 global step 的若干 micro-batches；
- 保证每个 denoising trajectory 内部策略一致性（trajectory-level consistency），防止 credit assignment 混乱。

> ✅ 优势：消除最终 micro-batch 后不可避免的同步等待 bubble，进一步压榨吞吐。

#### （4）**Bubble-Triggered TAG Search (BTS)**
- 动态搜索最优 TAG 插入策略：在每个出现的 bubble 处决定是否插入 TAG micro-batch；
- 采用递归搜索 + 分支剪枝 + 状态压缩，实现实时在线决策，避免局部最优。

> ✅ 优势：自适应应对运行时波动（如硬件延迟、冷启动等），最大化 end-to-end 效率。

---

### 相比现有方法的优势
| 特性 | veRL / GenRL | veRL-Omni | DigenRL |
|------|--------------|-----------|---------|
| Diffusion-RL 支持 | ❌ | ✅（有限） | ✅（完整支持） |
| 解耦架构（Disaggregated） | ❌ | ❌ | ✅ |
| 细粒度流水线 | ❌ | ❌ | ✅（GAP + TSP） |
| 异构 GPU 支持 | ❌ | ❌ | ✅ |
| 弹性资源利用 | ❌ | ❌ | ✅（TAG） |
| 异步优化 | ⚠️（文本适用） | ❌ | ✅（TCSS，保持轨迹一致） |

> 🔍 DigenRL 是首个专为 **Diffusion-based 视觉生成模型**设计的高性能解耦 RL 框架。

---

## 2. 核心实验方法和设置

### 使用的模型与测试平台
#### 模型（Visual Generative LLMs）
- **HunyuanVideo-13B**（视频生成）
- **Wan2.1-14B**（视频生成）
- **FLUX.1-12B**（图像生成）
- **QwenImage-20B**（多模态图像生成）

#### 测试床（Testbeds）
| 名称 | 规模 | 网络 | GPU 类型 | 显存 | 性能 |
|------|------|--------|----------|-------|--------|
| Testbed A | 32× GPU | 200 Gb/s | 第三代 Tensor Core | 48GB | 38.7 FP32 TFLOPS |
| Testbed B | 32× GPU | 400 Gb/s | 第四代 Tensor Core | 80GB | 67 FP32 TFLOPS |
| Testbed C | 16× GPU | 200 Gb/s | 第五代 Tensor Core | 96GB | 126 FP32 TFLOPS |

> ✅ 包含同构与**异构混合集群**（A+C 联合使用）

---

### 实验设置与评估指标

#### 主要算法
- **FlowGRPO**（用于 QwenImage）
- **DanceGRPO**（用于 HunyuanVideo）

#### 训练配置
- Mini-batch size: 32
- Micro-batch size: 可变（由 GAP 控制）
- Sampling steps: 20 / 30 / 40
- Timestep fraction for training: 0.4 / 0.6
- 分布式后端：FSDP（训练）、vLLM-Omni（推理）

#### 评估指标
- **End-to-end training time per iteration**
- **Throughput**（单位时间内完成的 iterations 或 samples）
- **Speedup**（相对于 baseline 的加速比）
- **Ablation study**：逐项验证 GAP、TSP、TAG、TCSS 的增益
- **Reward curve stability**（验证 TCSS 不影响收敛性）

---

### 基线方法对比
| Baseline | 描述 |
|--------|------|
| **veRL-Omni** | 当前最先进的开源 Diffusion-RL 框架，采用 colocated 架构 |
| **Native Colocation** | 原生集成生成与训练的实现方式（如官方训练脚本） |
| **GenRL** | 支持多种视觉生成模型的 RL 框架，但系统级优化较弱 |

> ⚠️ 所有对比均保持相同模型、算法、backend（FSDP/vLLM-Omni），确保公平。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Speedup）

| 模型 | vs. veRL-Omni | vs. Colocated | vs. GenRL |
|------|----------------|------------------|-------------|
| **QwenImage-20B** | **1.56× ~ 2.21×** | — | — |
| **HunyuanVideo-13B** | — | **1.04× ~ 1.49×**（TCSS 开启） | ≈ GenRL（优于 colocation） |
| **Wan2.1-14B** | — | **1.13× ~ 1.45×**（TCSS 开启） | > GenRL（接近理论上限） |
| **FLUX.1-12B** | — | **1.06× ~ 1.21×** | > GenRL |

> 📈 最高达到 **2.21× 加速比**，平均提升 **1.56–2.10×**。

---

### 与基线方法的对比结果
- 在所有测试场景下，**DigenRL 显著优于 veRL-Omni 和原生 colocated 实现**；
- 在高 timestep 数（如 40 steps）和高训练比例（0.6）下增益更大，说明 TSP 对长序列训练更有效；
- 在异构环境下（Testbed A + C），DigenRL 实现 **1.46× → 1.85× 提升**，远超 colocated 的 1.46×，表明其对异构资源调度更具优势。

---

### 消融实验结果（Ablation Study）

#### HunyuanVideo 上的组件贡献（Figure 17）
| 阶段 | Speedup（vs. Colocated） | 增益来源 |
|------|--------------------------|----------|
| Disaggregated Only | 1.52× | 架构解耦基础 |
| + GAP | 1.84× | 更细粒度流水线 |
| + TSP | 2.14× | 训练效率提升 |
| + TAG | 3.05× | 利用 trainer 空闲资源 |
| + TCSS | 3.34× | 消除同步 bubble |

> 💡 TAG 贡献最大（+0.9×），TCSS 补充剩余瓶颈。

#### Wan2.1-14B 上的结果（Figure 18）
| 阶段 | Speedup |
|------|--------|
| Disaggregated | 1.58× |
| + GAP | 1.70× |
| + TSP | 2.10× |
| + TAG | 2.79× |
| + TCSS | 2.85× |

> ✅ 验证各模块正交且可叠加。

---

## 4. 关键结论和发现

### 主要发现
1. **解耦架构对 Diffusion-RL 至关重要**：尽管 rollout 工作负载相对均匀，但生成与训练的计算特性差异（sequential vs. parallel）导致强烈负载失衡，必须通过 disaggregation 实现灵活资源分配。
2. **GAP + TSP 实现细粒度并行**：突破传统 batch-level 并行限制，在小批量设置下也能实现高效流水线。
3. **TAG 显著缓解 generator 瓶颈**：trainer 的空闲周期可用于辅助生成，尤其适合 compute-bound 的训练阶段。
4. **TCSS 安全地打破同步壁垒**：通过控制 staleness ≤ 0.5 micro-batches，可在不影响 reward 曲线的前提下提升吞吐。
5. **BTS 实现动态最优调度**：递归搜索结合历史 profiling，能实时调整 TAG 策略，适应运行时变化。

> 📊 图16显示：TCSS（staleness=0.5）的 reward 曲线与 on-policy 几乎重合，证明其收敛稳定性。

---

### 方法的局限性
1. **依赖精确 runtime profiling**：BTS 的效果受生成/训练时间估计准确性影响，极端波动可能降低决策质量；
2. **context switching 开销**：trainer 执行 TAG 时需切换模型状态（offload/load），在频繁切换时可能引入额外延迟；
3. **目前仅支持单节点内 disaggregation**：跨数据中心或多租户场景尚未验证；
4. **对极短生成任务收益较小**：若 generation time 远小于 training，则 TAG 和 TCSS 增益受限。

---

### 未来工作方向
1. **自动化 hyperparameter tuning for TAG/TCSS**：基于 workload 自动调节 staleness ratio 和 micro-batch size；
2. **支持更多 Diffusion-RL 算法**：如 DDPO、MixGRPO 等；
3. **扩展至多模态 agent 训练**：结合 vision-language-action 的复杂任务；
4. **探索 zero-switching overhead 的统一模型架构**：减少 trainer/generator 切换成本；
5. **云原生部署优化**：结合 preemptible instances 和弹性伸缩，进一步降低成本。

---

> ✅ **总结一句话**：  
> **DigenRL 通过 diffusion-aware 的并行机制（GAP/TSP）、弹性资源利用（TAG）和安全异步同步（TCSS），首次实现了高性能、可扩展的解耦式 Diffusion-RL 框架，在多种视觉生成模型上取得 1.56–2.10× 的端到端加速，为下一代视觉生成智能体训练提供了基础设施支持。**

</details>

---

### 3. [BluTrain: A C++/CUDA Framework for AI Systems](https://arxiv.org/abs/2606.24780)

**Authors**: Adhitya Charan, Adwaid Suresh, Anuj Kumar, Aparna A, Dhanakumar K, Dharun M S, Dinesh G, Goutham Kumar Reddy K, Harshini V M, Jenifa D, Jona Delcy C A, Kathirvel S, Killi Uma Maheswara Rao, Kiruthik Kanna M, Kurra Vishnu Sai, Madhumithaa G K, Navin Kumar V, Ram Charan Golla, Revathi T, Rishikkanth R, Sanjay Krishna M V, Surendra Vendra  
**Category**: cs.AI  
**Published**: 2026-06-24  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.24780v1  

#### Abstract
Progress in deep learning is, at scale, more a matter of systems engineering than of modelling: the behaviour of a model in training (its throughput, its memory footprint, and the numerical fidelity of the result) is determined less by the architecture itself than by how that architecture is express...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《BluTrain: A C++/CUDA Framework for AI Systems》核心总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现代深度学习训练框架（如 PyTorch）在系统层面存在显著瓶颈，主要包括：
- **高开销抽象层**：Python 运行时、动态图调度、全局解释器锁（GIL）导致主机端延迟。
- **内存管理低效**：通用缓存分配器（如 PyTorch 的 `caching allocator`）易产生碎片，限制了大模型训练的可扩展性。
- **硬件表达不精确**：高层框架难以对算子进行细粒度控制，无法充分榨取 GPU 架构潜力（如 Tensor Core、SRAM 重用）。
- **分布式训练协调复杂**：通信-计算重叠、故障恢复、长上下文训练等需大量手动优化。

BluTrain 的目标是构建一个**从第一性原理出发**的训练框架，实现对硬件行为的绝对控制，同时消除系统复杂性，使建模过程无缝化。

---

### **提出的新方法与创新思路**

#### **(1) 全栈原生架构（Fully Native Stack）**
- **语言与依赖**：完全基于标准 C++ 和核心 CUDA 编程模型，无任何外部依赖（如 Python、cuDNN、cuBLAS）。
- **执行模式**：训练作为纯 C++ 进程运行，彻底绕过 Python GIL 和动态调度开销。

#### **(2) 自底向上协同设计（Co-Design from First Principles）**
所有组件均原生实现并紧密集成：
- **Tensor & Ops 模块**：带反向自动微分（reverse-mode autograd）的类型化张量系统。
- **BluBLAS**：手调 GEMM 库，专为 Ampere SM86 和 Ada Lovelace SM89 架构优化。
- **DTMS（Distributed Training Management System）**：统一管理 DDP、TP、CP 等并行策略。
- **MLIR-based Deep-Learning Compiler**：JIT 编译器，支持全局代数优化与 kernel 融合。
- **Deterministic Caching Allocator**：自适应块池分配器，最小化内部碎片。

#### **(3) 编译时特化（Compile-Time Specialization）**
- 所有算子通过 C++ 模板在编译期解析 dtype、layout、blocking、硬件架构等参数。
- 生成无分支、完全展开的机器码，消除运行时 dispatch 开销。

#### **(4) 分布式执行优化**
- **异步通信-计算重叠**：使用双 CUDA 流 + `cudaEvent` 实现 AllReduce 与 backward 计算并行。
- **Context Parallelism（CP）**：独立于 TP 的序列维度并行，支持超长上下文训练。
- **Ring Rotator Pipeline**：用于 KV 缓存跨设备传递，支持 All-to-All、P2P、AllGather 后端切换。

#### **(5) 故障容忍与弹性训练**
- 集成 **Orchestrator Daemon**，支持进程监控、健康检测、NCCL 错误恢复。
- 支持 ECC 错误、thermal excursion、OOM kill 等故障的自动重启与 checkpoint 恢复。

---

### **相比现有方法的优势**
| 维度 | PyTorch / Megatron-LM | BluTrain |
|------|------------------------|--------|
| **执行环境** | Python + C++ backend | 纯 C++ native runtime |
| **调度开销** | 存在 GIL 和动态 dispatch | 编译期特化，零运行时开销 |
| **内存效率** | 通用 size-class 分配器，易碎片化 | 动态对齐 + Oth 步缓存清理，减少 22% footprint |
| **数值保真** | 可能因近似算子引入误差 | 严格数学稳定算法，bit-exact 验证 |
| **可扩展性** | 依赖第三方库（如 NCCL, cuDNN） | 完全自主控制，性能天花板由自身决定 |

---

## **2. 核心实验方法和设置**

### **数据集**
- **FineWeb-Edu**：10B token 的高质量教育文本语料，用于训练 GPT-2 模型。

### **模型配置**
- **基准模型**：Decoder-only GPT-2，124M 参数（768 hidden dim, 12 layers, 12 heads）
- **长上下文变体**：context length 从 1024 扩展至 16,384
- **大模型测试**：2.42B 参数 GPT-2，在单卡上验证最大可训练规模

### **硬件平台**
- **主测试节点**：8× RTX 6000 Ada（48 GiB VRAM each）
- **辅助测试**：RTX 5070 双卡节点，用于 TP/CP 对比

### **评估指标**
| 指标 | 描述 |
|------|------|
| **Throughput (tok/s)** | 每秒处理的 token 数量 |
| **Memory Footprint (GiB)** | 单 GPU 峰值 VRAM 占用 |
| **Numerical Fidelity** | 最终 validation loss、训练轨迹一致性 |
| **Checkpoint Latency** | 保存/加载完整状态的时间 |
| **Scaling Efficiency** | 多卡下的加速比与通信隐藏能力 |

### **基线方法对比**
- **PyTorch (eager)**：标准动态图模式
- **PyTorch (compile)**：使用 TorchDynamo + Inductor 的编译模式
- **Megatron-LM**：用于 TP 性能对比
- **DeepSpeed-ZeRO/FSDP**：隐含对比对象（BluTrain 不依赖参数分片即可训练更大模型）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) 吞吐量（Throughput）**
| 方法 | 平均吞吐 (tok/s) | 步时延 (ms/step) |
|------|------------------|------------------|
| PyTorch (eager) | 394,663 | 1359.14 |
| PyTorch (compile) | 402,453 | 1326.73 |
| **BluTrain (eager)** | **406,595** | **1313.43** |

✅ **提升约 3%**，且在所有训练步骤中持续领先（见 Figure 4）。

#### **(2) 内存占用（Memory Footprint）**
| 方法 | Step 0 (GiB) | Train (GiB) | 减少比例 |
|------|-------------|------------|---------|
| PyTorch (default) | 24.09 | 27.54 | — |
| PyTorch (pow-2) | 28.04 | 26.35 | — |
| PyTorch (compile) | 20.22 | 24.38 | — |
| **BluTrain** | **23.01** | **21.48** | **↓22%** |

- 原因：确定性分配器 + 第0步后主动释放临时缓冲区（5–7% 内存回收）。

#### **(3) 数值保真度（Numerical Fidelity）**
| 方法 | 最小训练损失 | 最终验证损失 |
|------|--------------|----------------|
| PyTorch (eager) | 2.8793 | 3.0695 |
| PyTorch (compile) | 2.8765 | 3.0694 |
| **BluTrain** | **2.8771** | **3.0675** |

✅ 训练曲线几乎完全重合（Figure 3），最大偏差 < 3×10⁻³，收敛质量更优。

#### **(4) 长上下文训练（Long-context Training）**
| 指标 | PyTorch | BluTrain |
|------|--------|----------|
| context length | 16,384 | 16,384 |
| batch size | 2 | 2 |
| 峰值内存 (GiB) | 47.8 | 40.4 |
| 显存余量 (headroom) | 0.2 GiB (0.4%) | 7.6 GiB (15.8%) |
| 吞吐 (tok/s) | 14,849 | **17,784** |

✅ 在相同硬件下，BluTrain **多出 7.6 GiB 安全空间**，避免因碎片导致 OOM。

#### **(5) 最大可训练模型规模**
| 模型 | 参数量 | PyTorch | BluTrain |
|------|--------|--------|----------|
| GPT-2 variant | 2.42B | ❌ OOM | ✅ 成功训练 |
| 峰值内存 | — | N/A | 46.9 GiB |
| 吞吐 | — | — | 13.6K tok/s |

✅ 在单张 RTX 6000 Ada 上直接训练 2.42B 模型，无需模型并行或 offloading。

#### **(6) 分布式性能**
- **Tensor Parallelism (TP=2)**：
  - 相比 Megatron-LM 提升 **19.7%** 吞吐。
  - 异步双流协议（AsyncTP）进一步提升 ~7.9%。
- **Context Parallelism**：
  - 在 RTX 6000 Ada 上相较 PyTorch 提升 **+30.1%** 吞吐（Figure 40）。
  - 所有 rotator 后端（All-to-All/P2P/AllGather）均优于基线。

#### **(7) Checkpoint 性能**
| 模型 | 方法 | Save Staging Stall (ms) | Full Save (ms) |
|------|------|--------------------------|----------------|
| 124M | PyTorch | 360 | 360 |
| 124M | BluTrain (async) | **57** | 503（后台完成） |

✅ 异步 staging 将训练阻塞时间从 360ms 降至 **57ms**，其余写入在后台隐藏。

---

### **消融实验结果（Ablation Studies）**
尽管论文未提供系统性的模块级消融分析，但从以下方面可推断各组件贡献：
- **Allocator**：PyTorch 应用相同缓存清理策略反而导致内存回升（size-class 重新分配），说明 BluTrain 的确定性分配器是关键。
- **Compiler + Kernel Fusion**：微基准显示 Attention、GELU、LayerNorm 等 kernel 延迟全面低于 PyTorch（Appendix B）。
- **Async DDP**：bucket size 实验表明最优值在 25–50MB 区间，支持高效通信隐藏（Figure 35）。

---

## **4. 关键结论和发现**

### **主要发现**
1. **系统工程决定训练上限**：模型性能更多取决于“如何在硬件上表达”，而非架构本身。
2. **原生控制带来显著收益**：通过全栈自研 + 编译期特化，可在不牺牲灵活性的前提下获得 **3% 吞吐提升 + 22% 内存压缩**。
3. **内存效率直接影响可扩展性**：更小的 footprint 使得在单卡上训练更大模型成为可能（如 2.42B 参数）。
4. **数值保真是可复制训练的基础**：严格的数学稳定性确保训练轨迹可复现，收敛质量更高。
5. **长上下文训练需要专用并行机制**：Context Parallelism 结合 Ring Rotator 是突破序列长度瓶颈的有效路径。

---

### **局限性**
| 局限 | 说明 |
|------|------|
| **当前仅支持 NVIDIA GPU** | 未支持 AMD ROCm 或其他异构架构 |
| **缺乏跨架构泛化验证** | 实验集中在 Ada Lovelace 架构，未覆盖 Hopper/H100 等数据中心级芯片 |
| **缺少模块级消融分析** | 无法量化每个子系统（如 compiler、allocator）的具体性能贡献 |
| **分布式扩展限于单节点** | 多节点场景下的网络拥塞、链路退化尚未验证 |
| **故障预测仍待实证** | ECC、温度趋势预测阈值尚未基于大规模部署数据校准 |

---

### **未来工作方向**
1. **硬件无关化（Hardware Agnosticism）**
   - 将 MLIR 编译器扩展为通用 backend，支持 CUDA、ROCm、Triton、甚至 ASIC。
2. **数值精度建模**
   - 建立数学模型，预测不同算术格式（FP32/TF32/BF16）、累加顺序对最终 loss 的影响。
3. **多模态大模型训练**
   - 基于 BluTrain 原生性能优势，拓展至 NLP、CV、Speech、Multimodal 等领域。
4. **生产级容错增强**
   - 在真实数据中心环境中验证故障恢复机制，完善 predictive fault detection。
5. **开放生态探索**
   - 当前为闭源研究框架，未来可能开源部分组件（如 BluBLAS、DTMS）以推动社区发展。

---

> **总结一句话**：  
> **BluTrain 证明了“从第一性原理构建”的全栈原生训练框架，在吞吐、内存、数值保真三大维度上均可超越主流框架，为下一代 AI 系统提供了新的设计范式。**

</details>

---

### 4. [ModTGCN: Modularity-aware Graph Neural Networks for Text Classification](https://arxiv.org/abs/2606.23694)

**Authors**: Rajarshi Misra, Aditya Sharma, Vinti Agarwal, Hari Om Aggrawal  
**Category**: cs.CL  
**Published**: 2026-06-24  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.23694v1  

#### Abstract
Graph-based text classification models typically rely on local neighborhood aggregation and overlook global community structure, despite semantic document graphs exhibiting strong class-consistent clustering. Ignoring this can blur class boundaries and lead to over-smoothing. We propose ModTGCN, a m...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：ModTGCN: Modularity-aware Graph Neural Networks for Text Classification**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**
- 现有的基于 **Graph Neural Networks (GNN)** 的文本分类模型（如 TextGCN）主要依赖**局部邻域聚合**（local neighborhood aggregation），忽略了图中潜在的**全局社区结构**（global community structure）。
- 这导致两个关键问题：
  1. **类边界模糊**（Blurred class boundaries）：高频词（hub nodes）或噪声边可能导致不同类别节点被错误地拉近。
  2. **过平滑问题**（Over-smoothing）：深层 GNN 中节点表示趋于同质化，尤其在低同配性（low homophily）数据集中表现更差。

### 🚀 **提出了什么新方法或新思路**
提出 **ModTGCN** —— 一种**模块感知的图神经网络**（modularity-aware GNN），其核心思想是：
- 在标准 **cross-entropy loss** 基础上引入一个基于 **modularity** 的辅助目标函数，联合优化分类性能与社区一致性。
- 构建一个独立的 **document-document similarity graph**（$ \mathcal{G}_{\text{doc}} $），基于预训练或微调的 **transformer embeddings**（如 SBERT）计算相似度（使用 RBF 核）。
- 利用该图上的 modularity 作为正则项，鼓励形成**类一致的文档社区**（class-coherent communities）。

此外还提出以下技术改进：
- **架构解耦**（Architectural decoupling）：将原始异构图（document-word + word-word）拆分为两个独立子图，显著提升训练效率（2×–10× 加速）。
- **混合监督策略**（Hybrid supervision）：对有标签节点使用真实标签，无标签节点使用 TextGCN 预测的伪标签（pseudo-labels）来构建 modularity 矩阵。

### ⭐ **相比现有方法的优势**
| 方面 | 优势 |
|------|------|
| **性能** | 在多个基准数据集上优于 TextGCN、TensorGCN 等 GNN 方法，尤其在低同配性数据集（如 Ohsumed, 20NG）上提升显著。 |
| **效率** | 图结构解耦大幅降低每 epoch 计算开销，训练速度提升达 10 倍（尤其在大规模 20NG 上从 1450s → 150s）。 |
| **泛化能力** | 不依赖完整 LLM 微调，兼容多种 encoder（如 BERT, RoBERTa），具备良好扩展性。 |
| **鲁棒性** | 显式建模全局结构，缓解 hub 节点影响和过平滑问题。 |

---

## 2. **核心实验方法和设置**

### 📚 **使用了哪些数据集**
共五个标准文本分类 benchmark 数据集：
- **MR**（电影评论情感分析）
- **R8**, **R52**（Reuters 新闻分类）
- **Ohsumed**（医学文献主题分类）
- **20NG**（20 Newsgroups，新闻组话题分类）

这些数据集具有不同的 **homophily levels**（从高到低），用于验证方法在复杂结构下的有效性。

### 🛠️ **实验设置和评估指标**
- **预处理与划分**：沿用 TextGCN 的设置（滑动窗口大小=20，固定训练/验证/测试划分）。
- **模型配置**：
  - 隐藏层维度：200
  - 优化器：Adam，最多训练 300 轮，早停 patience=30
  - 超参数通过 **Optuna** 自动调优（learning rate, dropout, σ, γ, λ 等）
- **评估指标**：
  - 主要指标：**Micro-F1**（平均五次随机种子结果）
  - 辅助分析：Macro-F1、modularity 变化趋势、silhouette score（聚类质量）

### 🔁 **基线方法对比**
分为三类进行比较：

#### （1）纯 GNN 方法
- TextGCN
- TensorGCN
- WCTextGCN
- U-TextGCN

#### （2）基于 Transformer Embedding 的浅层模型
- Logistic Regression (LR) on pre-trained/fine-tuned SBERT
- Linear Probing（单层 MLP）

#### （3）大语言模型（LLM）零样本/少样本方法
- GPT-3（zero-shot 和 few-shot k=16）
- ChatGPT（few-shot k=2,5）

---

## 3. **主要实验结果和性能指标**

### 📊 **关键性能数据（Micro-F1 %）**

| Model | MR | R8 | R52 | Ohsumed | 20NG |
|-------|----|----|-----|---------|------|
| TextGCN | 76.74 | 97.07 | 93.56 | 68.36 | 86.34 |
| TensorGCN | 77.91 | 98.04 | 95.05 | 70.11 | 87.74 |
| **ModTGCN (P)** | **81.45** | **97.55** | **94.54** | **71.97** | **90.60** |
| **ModTGCN (F)** | **88.07** | **98.70** | **96.16** | **77.52** | **91.14** |

> 注：(P) = 使用预训练 embedding；(F) = 使用任务微调后的 embedding

#### ✅ 主要观察：
- 在所有数据集上均超越 TextGCN，最大提升出现在 **Ohsumed (+9.16)** 和 **20NG (+4.8)**。
- 即使不微调 embedding，ModTGCN(P) 也优于多数基线，说明 **modularity 正则项本身带来强归纳偏置**。
- 微调后性能进一步跃升，在 **R8 和 R52 上接近甚至超过 co-trained BERTGCN**。

### 🔍 **与 LLM 对比（Few-shot Setting）**

| Model | MR | R8 | R52 | Ohsumed | 20NG |
|-------|----|----|-----|---------|------|
| GPT-3 (k=16) | 89.15 | 91.58 | 91.56 | – | – |
| ChatGPT (k=5) | – | 82.43 | 90.13 | 45.39 | 47.05 |
| **ModTGCN (F)** | **88.07** | **98.70** | **96.16** | **77.52** | **91.14** |

➡️ 结论：尽管 GPT-3 在简单任务（如 MR）上略优，但在结构复杂的多类别任务（如 20NG）上，**ModTGCN 显著优于 few-shot LLM**，表明当存在显式图结构时，利用关系信息比提示工程更有效。

### 🔧 **消融实验结果（Ablation Studies）**

#### （1）不同 $ \mathcal{G}_{\text{doc}} $ 构造方式
| 方法 | 性能趋势 |
|------|--------|
| TF-IDF 内积 | 最差，缺乏语义建模能力 |
| Cosine 相似度 | 中等 |
| **Gaussian (RBF) kernel** | **最优**，局部亲和性更好捕捉语义邻域 |

#### （2）边重加权（Edge Reweighting）
- 同类邻居边 × α (>1)，异类 × β (<1)
- 结果显示：α≈1.2, β≈0.4 时效果最佳，**小幅注入标签先验可增强社区凝聚性**

#### （3）标签策略（Labeling Strategy）
| 策略 | 效果 |
|------|------|
| Gold labels（仅训练集） | 次优 |
| **Predicted labels（全部节点）** | **更优**，尤其在 Ohsumed 上 F1 提升 0.5+，说明伪标签鲁棒且有助于防止过拟合 |

#### （4）超参数敏感性分析**
- **损失权重 λ ∈ [0.25, 0.5]** 效果最好，说明需平衡分类与社区结构。
- 分辨率参数 γ ≈ 3.0 最佳，避免过度合并或分裂社区。
- 邻居数量 k ≈ 10 表现稳定。

---

## 4. **关键结论和发现**

### ✅ **主要发现**
1. **Modularity 是有效的全局正则化机制**：
   - 引导 GNN 学习符合类别的社区结构，弥补局部聚合的不足。
   - 特别适用于 **低同配性（low homophily）场景**，其中局部邻域不可靠。

2. **性能增益集中在结构复杂的数据集**：
   - 如 **Ohsumed (homophily=0.16)** 和 **20NG (0.19)** 上提升最大（+11.54 F1）。
   - 而在高同配性数据（如 MR=0.70）上增益较小，因 embedding 已近乎线性可分。

3. **解耦架构大幅提升可扩展性**：
   - 无需维护 word-node 中间状态，减少约 50% 边遍历操作。
   - 实际训练时间下降 **2× 至 10×**，尤其在 20NG 上从 1450s → 150s。

4. **伪标签鲁棒性强**：
   - 即使早期预测不准，modularity 仍能稳定收敛（warm-up 几乎无影响）。
   - 支持端到端联合训练，无需额外预热阶段。

### ⚠️ **方法的局限性**
- 依赖高质量的 document embedding 来构建 $ \mathcal{G}_{\text{doc}} $，若 embedding 质量差会影响 modularity 效果。
- 当前 modularity 计算仍基于静态图，未考虑动态演化过程。
- 对于极短文本（如 tweet），TF-IDF 和 PMI 构建的词图可能稀疏且噪声大。

### 🔮 **未来工作方向**
- 将 modularity 扩展至 **动态图学习** 或 **异构图表示学习**。
- 探索与其他社区检测算法结合（如 InfoMap、Leiden）。
- 应用于其他需要建模 mesoscopic structure 的任务：**topic modeling、document clustering、knowledge graph completion** 等。

---

> 📌 **一句话总结**：  
> **ModTGCN 通过引入 modularity 作为全局正则项，成功将社区结构先验融入 GNN 文本分类框架，在保持高效的同时显著提升了在复杂、低同配性数据上的性能，为结构感知的文本建模提供了新范式。**

</details>

---

### 5. [A Comparative Study of Bayesian Contextual Bandits for Real-Time Warehouse Sorter Optimization](https://arxiv.org/abs/2606.23977)

**Authors**: Tina Dongxu Li, Mouhacine Benosman, Ken Meszaros, Trevor Dardik  
**Category**: cs.LG  
**Published**: 2026-06-24  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.23977v1  

#### Abstract
Efficient sorter diversion control of automated material handling systems (MHS) is critical for optimizing operational efficiency in large-scale warehouse environments. In this study, we use an inbound receiving sorter at a high-volume e-commerce warehouse as our primary use case, where the sorter d...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Comparative Study of Bayesian Contextual Bandits for Real-Time Warehouse Sorter Optimization

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
本文针对**大型电商仓库中的自动化分拣系统（Automated Material Handling Systems, MHS）在动态环境下的实时分流控制优化问题**。具体而言：
- 当前的分拣器依赖于具有**静态权重配置的成本函数（cost function）**，无法适应高度动态的系统上下文（如流量模式、拥堵水平、设备状态、上下游依赖等）。
- 这导致次优决策，引发高循环率（recirculation）、局部拥堵和运营效率下降。

### 🚀 提出的新方法与新思路
提出并系统比较了三种混合机器学习框架用于**实时分拣器分流优化**，重点验证了 **Bayesian Contextual Bandits (BCB)** 在该场景下的优越性：
- 将传统的基于规则的静态成本函数优化问题，转化为一个**Context-Action-Reward**的强化学习建模问题。
- 引入 **physics-aware emulator** 来解决冷启动（cold-start）问题，生成高质量模拟数据以支持离线训练到在线学习的安全过渡。
- 构建闭环架构，实现**持续在线学习（continuous online learning）** 和自适应调优。

### 🔍 相比现有方法的优势
| 方面 | BCB 框架优势 |
|------|-------------|
| **模型能力** | 超越传统 OR 方法（如 MILP），能处理高维 context 和非线性关系；相比黑盒式 Deep RL 更样本高效且可解释性强。 |
| **实时性** | 推理延迟极低（毫秒级），满足高频实时控制需求；而 XGB+BO 平均需 18 秒/次推理。 |
| **探索-利用平衡** | 内置 Thompson Sampling 实现自然的 exploration-exploitation trade-off。 |
| **策略特性** | 行为类似 Bang-Bang 控制，具备时间最优性（time-optimal control），能快速纠正系统偏差。 |

---

## 2. **核心实验方法和设置**

### 📊 数据集来源与构建
- **数据来源**：使用一个高保真度的 **physics-aware emulator** 模拟真实仓库 inbound receiving sorter 的运行。
- **数据规模**：共收集 5000 个 `[context, action, reward]` 元组样本。
- **特征维度**：
  - `context` 维度 $d_c = 14$：包括吞吐量、上游扫描数、目的地满载率、再循环率、控制覆盖率等。
  - `action` 维度 $d_w = 6$：即成本函数中六个 cost factors 的权重。
  - 特征向量总维度（含交互项）：$1 + d_c + d_w + d_c \times d_w = 105$

> ⚠️ 注：因历史操作数据缺乏 weight 变化（静态配置），无法直接用于训练，故采用仿真器随机扰动 weight 配置来生成多样化训练数据。

### 🔧 实验设置
- **时间窗口设计**：
  - **Context**：决策时刻前的滚动回看窗口（rolling lookback）
  - **Action**：应用在未来 $\Delta t$ 时间段内的 cost weights
  - **Reward**：在同一 $\Delta t$ 区间内观测的综合绩效指标
- **训练/测试划分**：80%/20%，增量训练（从10%到100%数据量）
- **评估阶段**：
  - 离线评估 reward model 准确性
  - 使用 reward model 作为 surrogate 估计 policy 的预期 reward uplift

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **RMSE / MAPE** | 评价 reward model 的预测准确性 |
| **Projected Reward Uplift (%)** | 对比候选模型 vs 启发式 baseline 的奖励提升 |
| **Action Distribution** | 分析推荐动作的分布特性（是否果断、敏感） |
| **Inference Latency** | 单次推理耗时，衡量实时性 |
| **Sample Efficiency** | 学习曲线收敛速度，反映数据利用率 |

### 🆚 基线与对比方法
| 方法 | 简称 | 描述 |
|------|------|------|
| **Heuristic Baseline** | — | 固定权重配置，由领域专家通过反复调试确定的最佳静态策略 |
| **Linear Regression + Gradient Descent Optimization** | LR+GDO | 线性模型 + Lasso 正则化 + PyTorch 可微优化搜索 |
| **XGBoost + Bayesian Optimization** | XGB+BO | 树模型预测 reward + BO 求解最优 action |
| **Bayesian Contextual Bandits** | BCB | 贝叶斯线性回归建模 reward + Thompson Sampling 决策 |

---

## 3. **主要实验结果和性能指标**

### 📈 关键性能数据汇总

#### ✅ Reward Model 预测准确率（Test Set）

| Model | RMSE | MAPE | 改进幅度（vs Baseline） |
|-------|------|------|------------------------|
| **XGB+BO** | 0.0492 | 4.56% | RMSE ↓52.12%, MAPE ↓56.74% |
| **BCB** | 0.0519 | 4.96% | RMSE ↓49.50%, MAPE ↓52.98% |
| **LR+GDO** | 0.0566 | 5.48% | RMSE ↓44.87%, MAPE ↓48.09% |
| **Mean Baseline** | 0.1027 | 10.55% | — |

> 💡 结论：XGB+BO 在 reward 预测上略胜一筹，但 BCB 仍具强竞争力。

#### 🏆 投影奖励提升（Projected Reward Uplift）

| Model | Reward Uplift (%) |
|-------|-------------------|
| **XGB+BO** | 1.75% |
| **BCB** | **2.03%** ✅ |

> ✅ **BCB 实现最高 reward uplift，优于表现最好的 reward predictor（XGB+BO）**

### 🔍 动作分布分析（Action Distribution）

| 观察点 | 发现 |
|--------|------|
| **BCB 动作分布呈“U型”或双峰分布** | 表明其倾向于将关键 action（如 Action1, Action6）设为 0 或 1，体现**决断性策略** |
| **XGB+BO 更保守** | 动作集中在中间值，调整温和 |
| **类比控制理论** | BCB 的行为类似于 **Bang-Bang Control**，适合高频反馈控制系统 |

### ⚙️ 推理效率对比

| 方法 | 平均推理延迟 |
|------|--------------|
| **BCB** | **< 10ms**（毫秒级） |
| **XGB+BO** | ~18 seconds |

> ⚡ BCB 推理速度快近 **1800倍**，更适合实时控制场景。

### 🔬 消融实验与归因分析（Feature Importance & Sensitivity）

- **BCB 模型系数分析（Table II）**：
  - Top 特征多为 **context-action 交互项**（如 `[Upstream throughput] × [Volume priority weight]`）
  - 证明 BCB 成功捕捉了“不同系统状态下最优权重应变化”的核心假设
- **XGBoost + SHAP 分析（Table III）**：
  - 主要识别出 context 中的关键因素（如 Routing compliance, Destination Fullness）
  - 但未显式建模 action 如何随 context 变化，缺乏策略层面的上下文敏感性

---

## 4. **关键结论和发现**

### ✅ 主要发现
1. **BCB 是最适合实时分拣优化的框架**：
   - 尽管 XGB+BO 的 reward model 更精确，但 **BCB 实现了最高的 reward uplift（2.03%）**
   - 得益于其**闭环学习机制、探索-利用平衡、快速响应能力和低延迟推理**

2. **静态权重配置已不足以应对现代仓库复杂性**：
   - 动态 context 显著影响最优 weight 配置
   - 必须引入 context-sensitive 的自适应优化机制

3. **emulator 是解决 cold-start 的有效手段**：
   - 可安全生成多样化训练数据，避免对生产系统的干扰
   - 支持从 offline 到 online 的平滑过渡

4. **BCB 的策略具有时间最优性（time-optimal）特征**：
   - 类似 Bang-Bang 控制的行为使其能在最短时间内纠正系统偏移
   - 在高频控制任务中优于渐进式调整策略

### ⚠️ 方法局限性
- **依赖高质量 emulator**：若模拟器不能准确反映物理世界动态，则迁移效果受限
- **reward 设计为即时回报**：未考虑长期延迟 reward，可能忽略某些累积效应
- **当前仅优化一组 sorter**：扩展至全仓多 sorter 协同仍需研究

### 🔮 未来工作方向
1. **闭环仿真验证**：通过 repeated randomized trials 和统计显著性检验进一步确认 reward uplift
2. **部署试点（Pilot Deployment）**：在真实环境中进行 A/B testing 验证实际收益
3. **多智能体协同扩展**：研究多个 sorter 之间的联合 Contextual Bandit 控制
4. **引入 Safety Constraints**：加入安全边界保障 exploration 不引发系统异常
5. **融合更多传感器信号与时序建模**：增强 context 表达能力

---

## ✅ 总结一句话
> 本研究表明，**Bayesian Contextual Bandits (BCB)** 凭借其卓越的实时性、上下文敏感性和闭环学习能力，在解决大规模仓库分拣器动态权重优化问题上展现出显著优势，是迈向智能化、自适应物流控制系统的重要一步。

</details>

---

### 6. [Decentralised AI Training and Inference with BlockTrain](https://arxiv.org/abs/2606.24722)

**Authors**: Peter Toth  
**Category**: cs.AI  
**Published**: 2026-06-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.24722v1  

#### Abstract
Frontier AI training is increasingly shaped by access to dense, centrally controlled accelerator clusters. This creates a structural advantage for hyperscalers and large centralized laboratories, and makes open or independent AI efforts depend on scarce capital, privileged infrastructure, and data-c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Decentralised AI Training and Inference with BlockTrain**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前前沿 AI 模型训练严重依赖于集中式的高性能计算集群（如 hyperscaler 数据中心），这导致：
- 小型机构或独立研究者难以参与；
- 模型训练受制于资本、基础设施和地理位置；
- 全模型同步训练需要每个 worker 持有完整的模型参数和优化器状态，对消费级 GPU 不友好。

**核心瓶颈**：标准反向传播（backpropagation）是全局性的，早期层的梯度依赖所有后续层的计算，难以在去中心化网络中高效并行。

### **提出了什么新方法或新思路**
提出 **Spheroid BlockTrain** 协议，一种**基于块局部目标（block-local objective）的去中心化训练框架**，其核心思想是：

- 将模型划分为多个可独立训练的 **residual blocks**；
- 每个 block 被赋予一个从全局目标导出的**局部去噪目标（denoising objective）**，基于 DiffusionBlocks 思想；
- 各 worker 只需训练自己负责的 block，无需持有完整模型或 optimizer state；
- 在推理时通过组合各 block 构成完整模型，实现“训练去中心化，推理组合化”。

该方法遵循 **Objective-Hardware Alignment** 原则：训练目标的粒度与硬件网络单元（单个 worker 所能承载的 block）对齐。

### **相比现有方法的优势**
| 对比维度 | 现有方法（如 Federated Averaging, DiLoCo） | BlockTrain |
|--------|------------------------------------------|-----------|
| **Worker 内存需求** | 需要存储整个模型及 optimizer state | 仅需存储一个 block，内存降低约 500× |
| **通信对象** | 完整模型更新或梯度 | 局部 block 更新 |
| **训练单位** | 全模型副本（full-model replica） | 独立可训练 block |
| **适用设备** | 至少能容纳全模型的小型集群 | 消费级 GPU（几 GiB 显存即可） |
| **系统弹性** | 依赖稳定连接和同步机制 | 支持异步、部分参与、动态加入/退出 |

> ✅ **本质区别**：不是“将集中式训练包装成去中心化”，而是**改变学习单元本身为局部可优化块**。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **Byte-level WikiText**：真实文本语料，用于验证模型在自然语言统计建模上的能力。
- 字节级别处理避免了 tokenizer 的引入，更贴近底层生成任务。

### **实验设置**
#### **模型结构**
- 使用 **Transformer block** 作为基本单元；
- 模型被划分为 $ B $ 个 blocks，每个 block 包含若干层 Transformer；
- 示例配置：$ B=3 $, 每 block 4 层，总深度 ~12 层，隐藏维度 $ d=512 \sim 18432 $。

#### **训练目标**
- **Block-local denoising objective**：
  - 输入：上下文 $ x $、加噪的目标嵌入 $ z_\sigma = e_y + \sigma \cdot \epsilon $
  - 输出：预测原始 token 分布
  - 使用 **EDM preconditioning**, **weighted cross entropy**, **lognormal sigma partitioning**
- 推理采用 **Euler denoising in embedding space**

#### **去中心化模拟设置**
1. **Single-GPU 异步模拟器**：
   - 模拟多个虚拟 worker，各自训练一个 block；
   - 引入 compute time、latency、bandwidth、dropout 等现实因素；
   - 设置 deadline 控制更新接受窗口。

2. **共享多卡训练（8xA100）**：
   - 6 个物理 worker 分配到 3 个 blocks（每 block 2 replica）；
   - Coordinator 负责聚合更新、组装模型、评估性能。

3. **HTTP/TCP 运输验证**：
   - 替换本地文件系统为网络 API；
   - worker 通过 HTTP 下载 checkpoint，上传更新；
   - 测试同一节点内和跨公网 IP 的传输可行性。

4. **分布式推理测试**：
   - 在三个公网 GPU 主机上部署 block stack；
   - 测试 **one-sweep serving** 路径：一次 traversal 完成整个序列输出。

### **评估指标**
| 指标 | 描述 |
|------|------|
| **Cross Entropy (CE)** | 主要训练质量指标 |
| **Perplexity (PPL)** | CE 的指数形式，越低越好 |
| **Participation Rate** | 异步场景下按时提交更新的比例 |
| **Network Traffic (GB)** | 应用层传输的数据量 |
| **Effective tok/s** | 推理吞吐量（有效 token/秒） |
| **Active AdamW Memory** | worker 上实际占用的 optimizer 内存量 |

### **基线方法对比**
- **End-to-end Transformer**：相同设置下的集中式全模型训练，作为性能上限参考。
- **Plain autoregressive TCP pipeline**：传统逐 token 生成的 WAN 推理基线，用于对比 serving 效率。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

| 实验设置 | Final CE | PPL | 备注 |
|---------|----------|-----|------|
| **BlockTrain Long Run** | **1.359** | **3.89** | $ B=3 $, 4-layer/block, 32 rounds |
| **End-to-End Transformer Ref.** | ~1.32 | ~3.74 | 相同 byte-level 设置 |
| **Shared 6-worker Run** | 1.385 | 3.99 | 多 worker 平均更新，训练一个组装模型 |
| **HTTP/TCP Transport (same-node)** | 1.576 | 4.83 | 移动 10.23 GB 数据，全参与 |
| **Public-IP WAN Run (larger)** | **1.811** | 6.11 | 从 5.580 改善，移动 15.22 GB |
| **One-Sweep Serving (max size)** | — | — | 达到 **75.80B 参数逻辑 fp16 形状** |

> 🔺 **与基线差距**：BlockTrain 最终 CE 仅比 end-to-end 高 **~0.04**，表明局部目标逼近全局性能。

### **与基线方法的对比结果**
- **vs. End-to-End Transformer**：
  - 性能接近（差 0.04 CE），但 worker 显存消耗极低（**< 0.2 GiB AdamW vs. 100.54 GiB 全栈**）；
  - 实现了“几乎无损”的去中心化训练近似。

- **vs. Plain Autoregressive Pipeline**（推理）：
  - BlockTrain：**一次 traversal 输出完整序列**；
  - AR Pipeline：**每个 token 都需一次 WAN traversal**；
  - 结果：BlockTrain 在长输出场景下吞吐高数十倍（如 64–128 token 输出时达 376 tok/s vs. 2.8 tok/s）。

### **消融实验结果**
#### **(1) Sigma Range 消融（表1）**
| Sigma Range | Final CE | 是否崩溃 |
|------------|----------|----------|
| [0.1, 2.0] | 14.295 | ❌ 推理完全失败（target leakage） |
| [1.0, 10.0] | **2.755** | ✅ 稳定且最优 |

> 📌 发现：**低 sigma 导致局部 CE 虚假优异，但破坏 from-noise inference**；必须设置较高最小 sigma（≥1）以防止信息泄露。

#### **(2) Block Expressivity 消融（表2 & 表3）**
| Block 类型 | Final CE | 解释 |
|-----------|----------|------|
| MLP Block | 2.740 | 目标可行，但表达力不足 |
| Transformer Block | **1.359** | 加入自注意力后显著提升 |

#### **(3) 固定总深度扫参（表3）**
| Config ($B$, $L_b$) | Final CE | Active AdamW |
|---------------------|----------|-------------|
| $B=12, L_b=1$ | 2.728 | 0.057 GiB |
| $B=3, L_b=4$ | **2.196 → 1.359 (long run)** | 0.198 GiB |

> 📌 发现：**质量随每 block 容量单调上升**，说明 local predictor 必须有足够的表达能力（GLN-style 观察）。

#### **(4) 异步参与率影响（表4 & 表5）**
| 场景 | Participation | Final CE | 结论 |
|------|---------------|----------|-------|
| Async Mild | 90% | 2.151 | ✅ 可容忍轻微延迟 |
| Async WAN | 42% | 2.595 | ⚠️ 质量下降但仍学习 |
| Async Bad Edge | 0% | 5.638 | ❌ 无更新即无学习 |

> 📌 关键发现：**系统边界由及时参与率决定**；只要 deadline 设置合理使参与率 >85%，质量可恢复至 lockstep 水平。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **Block-local denoising objective 是可行的去中心化训练路径**：
   - 在真实文本上达到接近 end-to-end 的性能（CE 差 < 0.04）；
   - 每 worker 仅需训练一个 block，显存开销极小（几 GiB GPU 可参与）。

2. ✅ **去中心化训练可在真实网络中运行**：
   - HTTP/TCP 协议已支持跨公网 IP 的 block checkpoint 和 update 传输；
   - 多 worker 可协作训练单一组装模型（非独立种子）。

3. ✅ **推理具有优越的 traversal economics**：
   - **One-sweep serving** 架构使得每次 WAN traversal 输出整段序列；
   - 相比 per-token traversal 的 autoregressive pipeline，带宽利用率大幅提升。

4. 🔍 **系统瓶颈不在带宽而在参与率**：
   - 质量取决于有多少更新能在 deadline 前到达；
   - 设计重点应是**如何激励及时提交**而非极致压缩通信。

5. 💡 **Objective-Hardware Alignment 是关键原则**：
   - 当训练目标的粒度与硬件能力匹配时，去中心化才真正可行。

---

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **尚未支持大规模扩展** | 当前实验集中在 ~12 层、数十亿参数规模，更大模型的行为未知 |
| **依赖特定目标设计** | 成功依赖 DiffusionBlocks 的 denoising 轨迹构造，其他架构迁移性待验证 |
| **缺乏 incentive 和 verification 层** | 当前协议假设诚实参与者，未解决恶意更新或伪造贡献问题 |
| **训练效率仍低于集中式** | 尽管内存友好，但整体收敛速度未超越 data-parallel 训练 |

---

### **未来工作方向**
1. **构建完整的去中心化协议栈**：
   - 添加 incentive layer（类似 Bittensor）、verification（如 zk-proofs）、stake/slash 机制；
   - 支持 permissionless participation。

2. **探索更高效的 block composition 方式**：
   - 动态路由、adaptive refinement、multi-path denoising。

3. **扩展到更大模型和更多任务**：
   - 如 vision-language、代码生成等；
   - 验证在 trillion-token 级别下的稳定性。

4. **优化异步调度策略**：
   - 自适应 deadline 调整；
   - 基于历史表现的 worker 权重分配。

5. **结合通信压缩技术**：
   - 将 DeMo/DisTrO 的 momentum compression 应用于 block update 传输，进一步降低成本。

---

> 🎯 **总体评价**：  
> BlockTrain 提供了一条**切实可行的技术路径**，让消费级硬件能够参与前沿 AI 模型的训练与服务。它不追求完全替代数据中心训练，而是开辟了一个新的可能性空间——**开放、协作、抗审查的 AI 生产模式**。虽然尚处早期，但已在真实文本上证明了其学习路径、传输路径和服务路径的完整性。

</details>

---

### 7. [An Efficient Construction of Completely Independent Spanning Trees in Dense Gaussian Networks](https://arxiv.org/abs/2606.23935)

**Authors**: Zaid Hussain, Fawaz AlAzemi, Bader AlBdaiwi  
**Category**: cs.DC  
**Published**: 2026-06-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.23935v1  

#### Abstract
Fault tolerance in routing and broadcasting is a critical aspect in ensuring the reliability and robustness of communication networks, particularly in environments prone to failures. This work presents an efficient method for constructing Completely Independent Spanning Trees (CISTs) within dense Ga...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：An Efficient Construction of Completely Independent Spanning Trees in Dense Gaussian Networks

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文致力于解决**在 Dense Gaussian Networks 中高效构建 Completely Independent Spanning Trees (CISTs)** 的问题。CISTs 在容错路由（fault-tolerant routing）和广播（broadcasting）中具有重要作用，能够提供多条节点不相交且边不相交的路径，从而提升通信网络的可靠性与鲁棒性。

然而，在一般图中构造两个或更多 CISTs 是一个 **NP-complete 问题**，因此需要针对特定拓扑结构设计高效的专用算法。

---

### 🚀 提出的新方法与创新点
1. **基于子集划分与旋转的 CIST 构造框架**  
   - 将 Dense Gaussian Network 划分为 **10 个互不重叠的节点子集**（如 $R, S, B1, D1$ 等），并据此定义第一棵 CIST（CIST1）的连接规则。
   - 第二棵 CIST（CIST2）通过将 CIST1 进行 **一次复数单位 $i$ 的旋转操作** 得到，利用了 Gaussian 整数格点的代数对称性。

2. **提出两种高效构造算法**
   - **Sequential Construction Algorithm**：时间复杂度为 $O(n)$，通信复杂度为 $O(d)$，其中 $n$ 为节点数，$d$ 为树深。
   - **Parallel Construction Algorithm**：时间复杂度为 $O(1)$，通信复杂度也为 $O(1)$（忽略初始广播开销 $O(k)$），适合大规模并行实现。

3. **理论深度优化**
   - 所构造的 CISTs 具有统一的深度：**$d = 3k - 1$**，其中 $k$ 是网络直径。
   - 显著低于 Pai et al. [15] 方法中的最大深度（例如当 $k=9$ 时，从 72 降至 23）。

---

### 🔍 相比现有方法的优势
| 维度 | 本文方法 | Pai et al. [15] |
|------|----------|----------------|
| **树深度** | $3k - 1$ | $\sim n/2 = k^2 + k + 0.5$ |
| **构造方式** | 子集划分 + 几何旋转 | 基于 Hamiltonian Cycle 分割 |
| **根节点要求** | CISTs 可拥有不同根节点（符合 CIST 定义） | 需共享根节点？ |
| **效率** | 更低深度 → 更短平均跳数、更快消息传递 | 较高深度导致更高延迟 |
| **可扩展性** | 支持并行化构造，适用于大型片上网络 |

> ✅ **优势总结**：本方法不仅更高效，而且生成的 CISTs 拥有显著更低的深度，提升了路由性能，并天然支持容错通信。

---

## 2. 核心实验方法和设置

### 📊 数据集 / 网络规模
实验在一系列 **Dense Gaussian Networks** 上进行，参数形式为 $\alpha = a + bi$，满足 $b = a + 1$，直径 $k = a$。具体测试的网络包括：
- $3+4i$, $4+5i$, $5+6i$, ..., $12+13i$
- 对应节点数量 $N(\alpha) = a^2 + b^2 = 2k^2 + 2k + 1$

> 示例：当 $k=9$ 时，网络含 $2×81 + 18 + 1 = 181$ 个节点。

---

### ⚙️ 实验设置
- **仿真工具**：使用 Python 的 **NetworkX** 库建模与分析图结构。
- **链路假设**：所有链路为全双工（full-duplex），允许同时收发。
- **故障模型**：考虑三种场景：
  1. 无故障节点（No Faulty）
  2. 单个故障节点（1 Faulty Node）
  3. 两个故障节点（2 Faulty Nodes）
- **评估策略**：
  - 使用暴力枚举法遍历所有可能的故障位置组合。
  - 在每种情况下计算从根节点到其他所有节点的最长路径长度（即“最大步数”）。
  - 最终报告 **平均最大步数（average maximum number of steps）**。

---

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Tree Depth** | CIST 的最大高度（从根到最远叶节点的距离） |
| **Average Maximum Steps** | 跨越所有目标节点和故障配置下的平均最长路径长度 |
| **Improvement (%)** | 相对于基线方法的性能提升百分比 |

---

### 🔀 基线方法对比
- 主要对比对象：**Pai et al. [15]** 提出的 CIST 构造算法
- 对比维度：树深度、平均最大步数、容错能力

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Tables 6–9 和 Fig. 4–7）

#### 表：不同网络规模下的平均最大步数（部分摘录）

| $\alpha$ | No Faulty | 1 Faulty | 2 Faulty |
|---------|-----------|----------|----------|
| 3+4i    | 2.40      | 3.72     | 3.26     |
| 6+7i    | 4.69      | 8.20     | 7.77     |
| 9+10i   | 6.73      | 13.19    | 12.70    |
| 12+13i  | 9.04      | 17.63    | 17.16    |

> 注：随着网络增大，平均步数线性增长，远优于基线的二次增长趋势。

---

#### 性能提升对比（vs. Pai et al. [15]）

| 场景 | 最小提升 | 最大提升（大网络） |
|------|--------|------------------|
| **无故障** | 50.31% ($k=2$) | **82.85%** ($k=12$) |
| **单故障** | 26.56% ($k=2$) | **67.27%** ($k=12$) |
| **双故障** | 33.64% ($k=2$) | **67.62%** ($k=12$) |

> 💡 结论：网络越大，优势越明显；即使存在故障，仍保持显著性能增益。

---

#### 树深度对比（Table 5）

| $k$ | Pai et al. [15] | 本文方法 | 提升幅度 |
|-----|------------------|------------|-----------|
| 5   | 20               | 11         | ~45% ↓     |
| 8   | 56               | 20         | ~64% ↓     |
| 9   | 72               | 23         | ~68% ↓     |

> ✅ 所提方法实现了 **至少 33% 的改进**（摘要中声称），实际最高达 **近 70%**。

---

### ❌ 消融实验（Ablation Study）
论文未明确进行传统意义上的消融实验（如移除某模块验证影响），但通过以下方式间接体现设计有效性：
- 不同子集划分规则的合理性证明（Lemmas 1–5）
- 旋转机制带来的第二棵独立树的有效性验证
- 多种故障模式下性能一致性表明结构稳健

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Dense Gaussian Networks 中最多可构造 2 个 CISTs**，本文方法达到该上限。
2. 所提出的 **子集划分 + 旋转构造法** 是系统化、可证明正确的，且具备良好几何直观性。
3. 构造出的 CISTs 深度仅为 $3k - 1$，远小于已有方法（接近 $k^2$ 量级），极大降低通信延迟。
4. 实验表明，无论是否存在故障节点，所提方法均显著优于基线，**提升幅度随网络规模增加而扩大**。
5. 并行算法可在常数时间内完成构造，极具实用价值，尤其适用于 SoC 或数据中心片上网络。

---

### ⚠️ 方法的局限性
1. **仅适用于 Dense Gaussian Networks**，难以直接推广至其他拓扑（如 Torus、Mesh、Hypercube）。
2. **最多只能构造 2 个 CISTs**，受限于图的最大边不相交生成树数量。
3. **依赖全局拓扑知识**（每个节点需知道自身坐标及网络参数 $\alpha$），对动态网络适应性较差。
4. 未讨论动态更新或增量重构的能力（如节点加入/退出）。

---

### 🔮 未来工作方向
1. 探索在 **非密集型 Gaussian Networks** 或其他代数图类中应用类似构造思想。
2. 设计支持 **更多 CISTs** 的新型拓扑结构或编码方案。
3. 开发 **分布式自适应版本** 的构造算法，减少对全局信息的依赖。
4. 将 CISTs 应用于 **安全路由、多播协议、负载均衡** 等高级应用场景。
5. 结合硬件实现（FPGA/GPU）进行真实环境部署与性能验证。

---

## ✅ 总结一句话
本文提出了一种基于子集划分与复平面旋转的高效 CIST 构造方法，在 Dense Gaussian Networks 上实现了 **深度更低、容错更强、性能更优** 的完全独立生成树构建，相较现有方法在平均最大步数上取得 **高达 83% 的提升**，为高性能互连网络中的可靠通信提供了强有力的支持。

</details>

---

### 8. [AsyncOPD: How Stale Can On-Policy Distillation Be?](https://arxiv.org/abs/2606.24143)

**Authors**: Wonjun Kang, Kevin Galim, Seunghyuk Oh, Minjun Kang, Sanghyun Park, Donghoon Kim, Minjae Lee, Minseo Kim, Rishabh Tiwari, Yuchen Zeng, Hyung Il Koo, Kangwook Lee  
**Category**: cs.LG  
**Published**: 2026-06-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.24143v1  

#### Abstract
On-policy distillation (OPD) trains a student on its own rollouts guided by teacher feedback and is becoming increasingly important for large language model (LLM) post-training. Like reinforcement learning (RL), however, OPD faces an on-policy systems bottleneck, as rollouts can dominate training ti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：AsyncOPD: How Stale Can On-Policy Distillation Be?**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
本论文系统研究了**异步 On-Policy Distillation (OPD)** 中的**策略陈旧性（staleness）问题**。在大规模语言模型（LLM）后训练中，OPD 通过学生模型的 rollouts 并结合教师模型的 token-level 反馈进行训练，但其面临与强化学习（RL）类似的“on-policy 系统瓶颈”——每次 learner 更新必须等待新的 rollouts 完成，导致训练效率低下。

为提升效率，可采用异步训练流水线（asynchronous pipeline），将 rollout 生成、教师打分和 learner 更新解耦。然而，这会引入**陈旧策略数据（stale-policy data）**，即 learner 使用的是旧版本学生策略生成的数据，可能导致训练不稳定或性能下降。

此外，由于教师模型的 full-vocabulary logits 存储和传输成本高昂，实际中通常只缓存有限动作上的教师分数（finite teacher-score cache），进一步加剧了估计偏差和方差问题。

### **提出了什么新方法或新思路**
论文提出并开源了 **AsyncOPD** ——一个完全异步的 OPD 训练框架，并围绕以下三个核心问题进行了系统性探索：

1. **KL 方向对陈旧性的鲁棒性差异**  
   首次揭示：**forward KL**（教师加权）对陈旧 rollouts 更鲁棒，而 **reverse KL**（学生加权）更脆弱，因其依赖当前学生的动作分布，易受缓存支持不匹配影响。

2. **针对 reverse KL 的最优 policy-gradient surrogate**  
   发现：在 reverse KL 下，最有效的策略是**在 learner 端重新计算当前学生的 advantage $A_\theta$，且不使用 clipping**。相比 PPO-style 或异步 RL 中的先进方法（如 Decoupled PPO、M2PO），该简单设计表现更优。

3. **多样本蒙特卡洛估计器（multi-sample MC）**  
   针对有限缓存带来的偏差-方差权衡，提出 **multi-sample Monte Carlo (MC)** 方法：在每个 decoding 步骤从行为策略中采样多个动作并缓存其教师分数。相比单样本 MC，它显著降低方差；相比 top-k 缓存，它保持无偏性（correctable via importance sampling）。

最终，基于上述分析构建了 **AsyncOPD 流水线**，实现了 rollout、teacher scoring 和 learner update 的完全重叠。

### **相比现有方法的优势**
- **更高吞吐量**：相比严格同步训练，吞吐提升 **1.6× 到 3.8×**。
- **维持模型质量**：在 AIME24、AIME25、AMC 等数学推理任务上达到与同步训练相当甚至更好的准确率。
- **工程实用性强**：无需存储 full-vocabulary 教师 logits，适合大规模部署。
- **理论指导明确**：提供了关于 KL 方向选择、surrogate 设计、缓存策略的清晰准则。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **训练数据**：`DeepMath` 数据集，筛选出难度 ≥6 的 57,630 个数学问题。
- **评估数据集**：
  - `AIME2024`（30题）
  - `AIME2025`（30题）
  - `AMC2023`（40题）

### **实验设置**
- **模型配置**：
  - **学生模型**：Qwen3-{1.7B, 4B, 8B}-Base
  - **教师模型**：Qwen3-30B-A3B-Instruct-2507
- **硬件平台**：单节点 8×B200 GPU
- **训练参数**：
  - Batch size: 256（mini-batch 64）
  - Optimizer: AdamW，LR = 3e-5，weight decay = 0.01
  - Rollout length: 2,048 prompt + 16,384 response tokens
  - Evaluation: 每题生成 32 个响应，取正确数平均
- **评估指标**：
  - **Avg@32**：每道题 32 个样本中的平均通过率（pass rate），用于衡量最终准确率。
  - **Training throughput (train tok/s)**：每秒处理的训练 token 数。
  - **Pipeline overlap**：衡量流水线并发程度（理想最大值为 3）。

### **基线方法对比**
- **Strict Sync**：严格同步 OPD，无陈旧性。
- **Two-step-off**：固定延迟两步的异步调度器（来自 VeRL [19]）。
- **AsyncOPD (ours)**：提出的完全异步流水线，支持动态队列深度控制。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 学生模型       | 调度器         | 方法   | 吞吐提升 (×sync) | 最终 Avg@32 |
|----------------|----------------|--------|------------------|-------------|
| Qwen3-1.7B-Base | Strict Sync    | MC64   | 1.00×            | 8.85        |
|                | AsyncOPD (ours)| MC64   | **2.70×**        | **9.38**    |
| Qwen3-4B-Base   | Strict Sync    | MC64   | 1.00×            | 25.00       |
|                | AsyncOPD (ours)| MC64   | **1.66×**        | 25.00       |
| Qwen3-8B-Base   | Strict Sync    | MC64   | 1.00×            | 26.56       |
|                | AsyncOPD (ours)| MC64   | **1.94×**        | 28.65       |

> 在 MC1 设置下，最高达 **3.28× 吞吐提升**（1.7B 模型）。

### **与基线方法的对比结果**
- **AsyncOPD 显著优于 Two-step-off 和 Strict Sync**：
  - 所有模型尺寸下均实现**最高吞吐量和 pipeline overlap**。
  - 在相同训练时间内达到更高的检查点，从而**更快收敛到高性能**（见图11）。
- **准确率保持竞争力**：
  - 多数情况下与同步训练持平或略优（如 1.7B 上从 8.85 → 9.38）。
  - 即使在高 staleness（如 128 步）下仍能稳定训练。

### **消融实验结果**
#### （1）Surrogate 设计对比（图3 & 表1a）
| Surrogate                     | 性能趋势 |
|-------------------------------|---------|
| $A_{\text{old}}$, clip        | 基线强，但随 staleness 下降快 |
| $A_{\text{old}}$, no clip     | 更差 |
| $A_{\theta}$, clip            | clip 损害信号，表现下降 |
| $A_{\theta}$, **no clip**     | ✅ **最佳，最稳定** |

> 结论：**recompute $A_\theta$ + no clip 是最优选择**。

#### （2）缓存策略对比（图6 & 图7）
| 方法             | 特点 | 准确率表现 |
|------------------|------|-----------|
| Top-k (stale)    | 固定支持，无法恢复缺失动作 | ❌ 支持不匹配，性能差 |
| Top-k + RW       | 重加权，但仍缺动作 | ❌ 无改善 |
| One-sample MC    | 可通过 IS 无偏修正 | ✅ 明显优于 top-k |
| **Multi-sample MC (m=4,16,64)** | 保持无偏性，降低方差 | ✅✅ **进一步提升大 staleness 下性能** |

> 表6 显示：当 $m=64$ 时，**局部方差降至单样本的 1.49%**，序列级方差降至 11.2%。

---

## **4. 关键结论和发现**

### **主要发现**
1. **KL 方向决定陈旧性鲁棒性**：
   - **Forward KL** 对 stale rollouts 更鲁棒（因教师加权）。
   - **Reverse KL** 更脆弱，但可通过 estimator 设计缓解。

2. **最优 reverse KL surrogate 是 $A_\theta$ + no clip**：
   - 优于 PPO-style、Decoupled PPO、M2PO 等复杂方法。
   - clipping 在 $A_\theta$ 场景下反而有害。

3. **Stale top-k 缓存存在根本性支持不匹配问题**：
   - 无法通过重加权修复缺失动作的教师分数。

4. **One-sample MC 可纠正但高方差，multi-sample MC 是理想折衷**：
   - 保持 importance sampling 的无偏性。
   - 通过多采样显著降低方差。

5. **AsyncOPD 实现高效异步训练**：
   - 吞吐提升 **1.6×–3.8×**。
   - 维持甚至提升最终准确率。
   - 支持 streaming 调度，最大化 pipeline overlap。

### **方法的局限性**
- **未支持 dense full-vocabulary KL**：受限于教师 logits 传输开销，目前仅适用于 sparse 或 sampled OPD。
- **实验局限于单节点**：尚未扩展至多节点集群，限制了最大规模验证。
- **假设行为策略覆盖当前策略支持**：若 $p_{\text{old}}$ 不覆盖 $p_\theta$，IS 将失效。

### **未来工作方向**
- 探索 **KDFlow 类方法在异步场景的应用**：通过传输 teacher hidden states 实现 dense KL，同时避免 full logits 传输。
- 扩展至 **multi-node 异步训练框架**，验证更大规模下的稳定性与效率。
- 研究 **自适应缓存机制**：动态调整 multi-sample 数量 $m$ 或缓存策略以平衡通信与计算成本。
- 探索 **staleness-aware learning rate 或 loss weighting** 机制，进一步提升高延迟下的鲁棒性。

---

> **代码地址**：[https://github.com/furiosa-ai/async-opd](https://github.com/furiosa-ai/async-opd)

</details>

---

### 9. [Qwen-AgentWorld: Language World Models for General Agents](https://arxiv.org/abs/2606.24597)

**Authors**: Yuxin Zuo, Zikai Xiao, Li Sheng, Fei Huang, Jianhong Tu, Yuxuan Liu, Tianyi Tang, Xiaomeng Hu, Yang Su, Qingfeng Lan, Yantao Liu, Qin Zhu, Yinger Zhang, Bowen Yu, Haiquan Zhao, Haiyang Xu, Jianxin Yang, Jiayang Cheng, Junyang Wang, Lianghao Deng, Mingfeng Xue, Tianyi Bai, Yang Fan, Yubo Ma, Yucheng Li, Zeyu Cui, Zhihai Wang, Zhihui Xie, Zhuorui Ye, An Yang, Dayiheng Liu, Jingren Zhou, Ning Ding  
**Category**: cs.CL  
**Published**: 2026-06-24  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.24597v1  

#### Abstract
A world model predicts environment dynamics based on current observations and actions, serving as a core cognitive mechanism for reasoning and planning. In this work, we investigate how world modeling based on language models can further push the boundaries of general agents. (i) We first focus on b...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Qwen-AgentWorld: Language World Models for General Agents 论文总结

## 1. 主要贡献和创新点

### 解决的问题
当前基于大语言模型（LLM）的智能体（Agents）研究主要集中在策略学习（policy learning），即从状态到动作的映射（states → actions）。然而，环境动态建模（world modeling），即预测在给定动作后环境如何变化（(states, actions) → subsequent states），被严重忽视。这种缺失限制了智能体的推理、规划和泛化能力。

本文提出，一个强大的通用智能体不仅需要决策能力，还需要具备对环境动态进行建模的能力。为此，作者系统地研究了如何构建和应用**语言世界模型**（Language World Model, LWM），以推动通用智能体的发展。

### 提出的新方法和新思路
论文提出了 **Qwen-AgentWorld**，这是首个能够跨多个领域进行统一环境模拟的原生语言世界模型（native LWM）。其核心创新点如下：

*   **统一的LWM基础模型**：Qwen-AgentWorld 是一个单一的语言模型，能够通过长链式思维（chain-of-thought）推理，模拟涵盖 **7个领域**（MCP, Search, Terminal, Software Engineering, Android, Web, OS）的智能体交互环境。这打破了以往为每个特定环境单独建模的范式。
*   **三阶段训练范式**：提出了“**CPT注入，SFT激活，RL锐化**”（CPT injects, SFT activates, RL sharpens）的端到端训练流程。
    *   **CPT (Continual Pre-Training)**：注入通用的世界知识和状态转移动力学。
    *   **SFT (Supervised Fine-Tuning)**：激活显式的“下一步状态预测”（next-state-prediction）推理模式。
    *   **RL (Reinforcement Learning)**：使用混合奖励（hybrid rubric-and-rule rewards）来锐化模拟保真度。
*   **两种互补的应用范式**：探索了LWM增强通用智能体的两种途径：
    *   **解耦（Decoupling）**：将LWM作为独立的**环境模拟器**（Environment Simulator），用于可扩展且可控的强化学习（Sim RL）。
    *   **统一（Unifying）**：将LWM作为**智能体的基础模型**（Agent Foundation Model），通过LWM预训练来提升下游智能体任务的性能。

### 相比现有方法的优势
*   **更全面的智能体架构**：首次将世界建模作为与策略同等重要的核心认知机制进行系统性研究和实现。
*   **更强的泛化能力**：单一模型覆盖多领域，避免了为每个任务定制模型的繁琐过程。
*   **更高的训练效率和可控性**：作为模拟器时，提供了真实环境无法比拟的可扩展性和可控性，能生成极端场景和针对性扰动来暴露智能体弱点。
*   **更优的性能上限**：作为基础模型时，证明了世界建模能力可以作为一种有效的“热身”，显著提升下游智能体的最终性能。

## 2. 核心实验方法和设置

### 使用的数据集
*   **训练数据**：收集了超过 **1000万条**来自7个领域的环境交互轨迹。数据来源包括：
    *   专用的智能体基础设施（如容器化沙箱、持久化终端会话）。
    *   公开的环境交互日志（如终端会话记录、开源工具调用日志）。
    *   内部积累的智能体轨迹。
*   **评估基准**：提出了 **AgentWorldBench**，这是一个综合性基准测试，用于评估LWM的质量。
    *   **构成**：基于5个前沿模型（如Claude Opus 4.6, GPT-5.4）在9个已建立的智能体基准（如Tool Decathlon, Terminal-Bench 1.0 & 2.0, OSWorld-Verified）上的真实交互构建而成。
    *   **特点**：完全**分布外**（out-of-distribution），确保评估的是泛化能力而非记忆能力。

### 实验设置和评估指标
*   **模型**：主要评估了两个规模的模型：`Qwen-AgentWorld-35B-A3B` 和 `Qwen-AgentWorld-397B-A17B`。
*   **评估协议**：采用开放式的五维评分标准（five-dimensional rubric），由LLM Judge对每条预测的环境观察进行打分。五个维度是：
    *   **Format** (格式)
    *   **Factuality** (事实性)
    *   **Consistency** (一致性)
    *   **Realism** (现实感)
    *   **Quality** (质量)
    最终得分为五个维度的平均值（范围0-100）。
*   **额外验证**：设计了基于规则的验证器（rule-based verifiers）进行确定性的检查，以补充开放式评分。

### 基线方法对比
与以下14种基线模型进行了对比：
*   **前沿闭源模型**：Claude Opus 4.8/4.6, Claude Sonnet 4.6, GPT-5.4, Gemini 3.1 Pro。
*   **开源权重模型**：DeepSeek-V4-Pro, Kimi K2.6, GLM-5.1, MiniMax-M2.7。
*   **通义千问家族模型**（无LWM训练）：Qwen3.5-35B-A3B, Qwen3.5-397B-A17B, Qwen3.6-Plus等。这些模型与Qwen-AgentWorld共享相同的架构，但没有经过CPT→SFT→RL的LWM专项训练，用于隔离LWM训练的效果。

## 3. 主要实验结果和性能指标

### 关键性能数据
*   在 **AgentWorldBench** 上，`Qwen-AgentWorld-397B-A17B` 取得了最高的总体平均分 **58.71**，超过了所有其他基线模型（GPT-5.4得分为58.25）。
*   在文本域（Text Domains）上优势明显，平均得分 **58.07**，尤其在Terminal（57.73）和SWE（68.49）任务上表现卓越。
*   在GUI域（GUI Domains）上，虽然Claude系列模型领先，但`Qwen-AgentWorld-397B-A17B`也取得了第五名的好成绩（59.69），证明了纯文本训练的LWM也能有效处理GUI状态。

### 与基线方法的对比结果
*   **超越前沿模型**：`Qwen-AgentWorld-397B-A17B`在总体和文本域上均优于GPT-5.4、Claude Opus等顶尖模型。
*   **验证LWM训练的有效性**：与同架构但未经LWM训练的基线相比，性能提升巨大。
    *   在397B规模下，总体平均分从 **54.74** 提升至 **58.71**（+3.97）。
    *   在35B规模下，提升更为显著，从 **47.73** 提升至 **56.39**（+8.66），甚至超过了Claude Sonnet 4.6。

### 消融实验结果
*   **跨域泛化**：仅在Terminal数据上进行RL训练，不仅能提升Terminal任务本身（+14.2），还能让未见过的MCP（+5.0）、SWE（+11.5）和Search（+11.8）任务性能同步提升。这证明了LWM学习到了可迁移的通用世界知识，而不仅仅是领域特定的输出格式。
*   **应用范式效果**：
    *   **作为环境模拟器**：在4000个OpenClaw环境中进行Sim RL，使`Claw-Eval`得分提升了+4.3，`QwenClawBench`提升了+7.1。可控模拟（如引入错误、部分结果）能带来远超非可控模拟的收益。
    *   **作为智能体基础模型**：仅用LWM RL进行单轮次、无工具调用的“热身”训练，就能在7个不同的下游智能体任务上取得显著提升，例如在`Claw-Eval`上提升了+11.3，在`BFCL v4`上提升了+9.0。这证明了世界建模能力具有极强的跨任务泛化能力。

## 4. 关键结论和发现

### 主要发现
1.  **世界建模至关重要**：语言世界模型不仅是有用的，而且对于构建真正的通用智能体可能是必要的。它为智能体提供了“预见”行动后果的能力。
2.  **LWM可作为强大基础**：通过专门的三阶段训练，可以成功构建高质量的LWM。`Qwen-AgentWorld`证明了单一模型可以有效地模拟复杂的多领域环境。
3.  **双重赋能路径**：LWM可以通过“解耦”和“统一”两种方式显著增强智能体。既能提供无限、可控的训练环境，又能直接提升智能体的决策上限。
4.  **涌现的元推理能力**：LWM训练使智能体内化了一种“预测驱动的行动优化”（prediction-driven action refinement）的元推理模式。分析显示，经过LWM训练的智能体在行动前会主动预测环境反馈，并据此修正计划。

### 局限性
*   **GUI领域的差距**：尽管在GUI任务上表现不俗，但其性能仍落后于经过多模态预训练的模型（如Claude, GPT-5），表明融合视觉信息是进一步提升的关键。
*   **幻觉风险**：作为生成模型，LWM在面对未知状态时可能产生合理的虚构，而非承认无知。虽然通过“认识边界意识”（epistemic boundary awareness）等机制缓解，但仍是挑战。
*   **计算成本**：三阶段训练，尤其是RL阶段，计算成本高昂。

### 未来工作方向
*   **智能体-LWM协同进化**（Agent-LWM Co-Evolution）：让智能体和世界模型相互促进，智能体探索新状态来挑战LWM，LWM则生成更难的场景来训练智能体。
*   **多模态扩展**（Multimodal Extension）：将GUI截图与文本状态表示融合，构建统一的视觉-语言世界模型。
*   **自适应仿真路由**（Adaptive Sim-to-Real Routing）：学习一个路由器，根据查询决定是调用世界模型还是真实环境，以平衡成本与保真度。
*   **动态工具合成**（Dynamic Tool Synthesis）：利用世界模型在运行时合成新的工具，而不是局限于预定义的工具集。

</details>

---

### 10. [Managing Task Execution for Unknown Workloads in Batteryless IoT: A Hardware-Agnostic Evaluation](https://arxiv.org/abs/2606.24340)

**Authors**: Samer Nasser, Henrique Duarte Moura, Ritesh Kumar Singh, Maarten Weyn, Jeroen Famaey  
**Category**: cs.LG  
**Published**: 2026-06-24  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.24340v1  

#### Abstract
In recent years, the Internet of Things (IoT) paradigm has been shifting toward batteryless, energy-harvesting architectures. Sustaining reliable operation in these systems requires intelligent management of highly volatile stored energy. As edge applications grow in complexity, traditional energy-a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Managing Task Execution for Unknown Workloads in Batteryless IoT: A Hardware-Agnostic Evaluation*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在 **batteryless IoT**（无电池物联网）系统中，设备依赖能量采集（如太阳能）和超级电容（supercapacitor）供电。由于环境能量高度波动且任务能耗未知（“black box”），传统的静态阈值调度策略（static thresholding）难以适应动态负载，容易导致系统宕机（power failure）或错失执行机会。

现有动态调度方法（如 AsTAR、模型驱动的动态阈值）通常依赖硬件特定参数或预知任务能耗，在实际部署中泛化能力差。

本论文旨在解决以下核心挑战：
- 如何在**不预先知道任务能耗**的情况下进行智能任务调度？
- 如何设计**硬件无关**（hardware-agnostic）的调度策略，适用于不同电容大小？
- 在复杂动态环境中如何平衡**任务吞吐量**与**系统生存能力**？

---

### 提出的新方法与创新思路

作者提出了两种全新的、无需先验知识的动态调度策略：

#### ✅ **Model-Free Reinforcement Learning (RL) Agent**
- 将任务执行决策建模为一个 **Markov Decision Process (MDP)**。
- 使用 **PPO (Proximal Policy Optimization)** 算法训练一个 RL agent，通过观察电压 $V$、采流 $I_H$、历史任务时长等状态来决定是否执行任务。
- 创新点：
  - **完全黑箱处理**：不需要任何任务能耗先验信息。
  - **域随机化（Domain Randomization）**：训练时随机改变电容大小（0.5F–10F），使策略具备跨硬件泛化能力。
  - **灵活奖励函数设计**：支持优化目标切换（如最小化 inter-task interval 或最小化 off-time）。

#### ✅ **Approximated Prediction (AP) Method**
- 动态估算当前任务的等效负载电阻 $R_{eq}$，用于预测执行后电容电压变化。
- 利用上一次任务执行前后的电压测量值，结合 Newton-Raphson 数值求解器反推 $R_{eq}$。
- 引入“预测-验证-校准”机制，在误差过大或发生宕机时触发重新校准。
- 创新点：
  - **轻量级实时估算**：无需离线训练或复杂模型。
  - **自适应安全边界**：根据估算电流动态调整 $V_{\text{min\_padded}}$，提升鲁棒性。
  - **低计算开销**：适合资源受限 MCU 部署。

---

### 相比现有方法的优势

| 方法 | 是否需任务能耗先验 | 是否硬件无关 | 可调性 | 计算开销 |
|------|---------------------|--------------|--------|----------|
| Static Thresholding | 否（但需手动调参） | ❌ | 低 | 极低 |
| AsTAR (AIMD/MIAD) | 否 | ✅ | 中 | 低 |
| Model-Based Dynamic Thresholding | ✅ | ❌ | 中 | 中 |
| **Proposed RL Agent** | ❌ | ✅ | ✅✅✅（高） | 高（需NN推理） |
| **Proposed AP Method** | ❌ | ✅ | 中 | 低（数值计算） |

> ✅ 表示优势明显；❌ 表示劣势或不满足

---

## 2. 核心实验方法和设置

### 数据集
- 使用来自 **冰岛 Hverageri 地区的真实太阳辐照数据**（2022年9月–2023年8月）。
- 原始采样频率为15分钟，通过线性插值得到 **30秒粒度的高分辨率数据**，共45天作为验证集。
- 转换为 harvesting current $I_H$，考虑了：
  - 太阳能板尺寸（60.1×41.3 mm）
  - 效率（18.5%）
  - PMIC效率（90%）
  - Cosine loss（25%）

### 实验设置
- **仿真平台**：基于物理精确建模的电容充放电方程（RC电路微分方程闭式解）。
- **系统组件模拟**：
  - MCU: STM32L4
  - 传感器: SHT30（温湿度）
  - LoRa模块: SX1262，支持 ADR（Adaptive Data Rate）
- **任务负载动态性建模**：
  - LoRa传输能耗随 RSSI 动态变化（映射 SF、TX Power、电流）
  - Payload 大小每日随机（20–255 bytes）
- **电容范围**：从 0.5F 到 10F（步进0.5F），测试硬件无关性。
- **唤醒周期**：每30秒唤醒一次，进行电压/电流采样并决策。

### 评估指标
| 指标 | 描述 |
|------|------|
| **Mean Daily Successful Executions** | 平均每天成功完成的任务数 |
| **Median Time Between Off-States** | 两次宕机之间的中位时间（越长越好） |
| **Mean Inter-Task Interval (ITI)** | 成功任务间的平均间隔时间 |
| **Median Daily Max ITI** | 每日最长任务间隔的中位数（反映夜间续航能力） |
| **Median Continuous Off-State Duration** | 宕机后恢复所需时间（越短越好） |

### 基线方法对比
| 方法 | 类型 |
|------|------|
| Static (1.9V) | 激进静态阈值（仅高于 $V_{\min}=1.8V$） |
| Static (3.45V) | 保守静态阈值（针对0.5F优化） |
| Optimal Static Threshold | 经过离线调优的最佳静态阈值 |
| AsTAR [5] | 自适应任务速率控制（AIMD/MIAD） |
| Short-Term Oracle (ST Oracle) | 理想上限：已知单次执行是否会宕机（不可实现） |
| Proposed: RL Agent (ITI-optimized / Off-Time-optimized) | 本文提出 |
| Proposed: Approximated Prediction (AP) | 本文提出 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Figures 5–9）

#### 📊 图5：Mean Daily Successful Executions
- **AP 方法表现最佳**，接近 **Short-Term Oracle** 和 **Optimal Static Threshold**。
- 在电容 ≥1.5F 时，AP 日均执行超 **1000次**。
- RL Agent（ITI优化版）也表现出色，略低于AP。
- AsTAR 执行频率最低，体现其“节制”特性。

#### 📊 图6：Median Time Between Off-States（生存时间）
- **RL Agent (Off-Time Optimized)** 和 **Static (3.45V)** 几乎完全避免宕机（≥2F时零宕机）。
- AP 和 ST Oracle 虽然执行多，但宕机更频繁 → “Boom and Bust”现象。
- Static (1.9V) 在小电容下生存极差。

#### 📊 图7 & 图8：Inter-Task Interval 分析
- **AsTAR 显著优于其他方法** 在 **最大 ITI** 上的表现：
  - 其他方法最大 ITI ≈ 8小时（对应夜间无法通信）。
  - AsTAR 在电容 >2.5F 时将最大 ITI 降至约 **1小时**，实现了**夜间持续通信能力**。
- 这表明 AsTAR 是唯一能有效“跨越长能量间隙”的方法。

#### 📊 图9：Recovery Time（宕机后重启时间）
- AP 和 RL Agent 在小电容（0.5–1F）下恢复更快。
- AP 在 1.5F–6F 区间恢复时间优于 Optimal Static，说明其**动态校准机制提升了韧性**。

---

### 与基线方法对比总结

| 方法 | 吞吐量 | 生存性 | 夜间续航 | 硬件无关性 | 适用场景 |
|------|--------|--------|-----------|------------|----------|
| **AP** | ✅✅✅（极高） | ✅ | ❌ | ✅✅✅ | 高频采集、无需夜间连续运行 |
| **RL Agent (ITI)** | ✅✅✅ | ✅ | ❌ | ✅✅✅ | 可配置优先级，追求高吞吐 |
| **RL Agent (Off-Time)** | ✅ | ✅✅✅ | ❌ | ✅✅✅ | 极端可靠性要求 |
| **AsTAR** | ❌ | ✅✅ | ✅✅✅ | ✅✅✅ | 必须保证夜间通信 |
| **Static (Optimal)** | ✅✅ | ✅✅ | ❌ | ❌ | 大电容、固定部署 |
| **Static (Aggressive)** | ✅✅ | ❌ | ❌ | ❌ | 不推荐 |
| **Static (Conservative)** | ❌ | ✅✅✅ | ❌ | ❌ | 牺牲性能保存活 |

---

### 消融实验结果（隐含分析）
虽然未明确标注“消融实验”，但文中通过以下方式进行了有效性验证：
- **RL Agent 的奖励函数影响**：比较 ITI vs Off-Time 优化版本，证明可通过 reward shaping 控制行为倾向。
- **AP 的校准机制必要性**：若不校准，预测误差累积会导致更多宕机。
- **域随机化对泛化的影响**：RL 在训练中引入电容变化，使其在测试中对所有电容尺寸均有效。

---

## 4. 关键结论和发现

### 主要发现
1. **没有一种方法在所有维度上都最优**：
   - 各方法有明确的**权衡取舍**（trade-offs）：吞吐量 vs 生存性 vs 夜间续航。

2. **硬件约束是选择策略的关键因素**：
   - 当电容较大（>5.5F）时，**简单的静态阈值即可达到接近最优性能**，无需复杂动态策略。
   - 当电容较小（<2F）时，**动态策略成为必需**，否则系统极易宕机。

3. **AP 方法实现了“近似Oracle性能 + 轻量化 + 硬件无关”三重优势**：
   - 在吞吐量上逼近理想上限（ST Oracle）。
   - 无需训练，部署简单，内存和计算开销远低于 RL。

4. **AsTAR 在“执行节奏控制”（pacing）方面无可替代**：
   - 唯一能在夜间维持通信的方法，特别适用于必须保持连接的应用（如安防监控）。

5. **RL 提供最大灵活性**：
   - 可通过 reward 设计定制行为（高频 or 高可靠）。
   - 但代价是更高的 MCU 资源消耗（Flash/RAM/能耗）。

---

### 方法的局限性
| 方法 | 局限性 |
|------|--------|
| **RL Agent** | - 推理阶段仍有可观的 energy/memory 开销<br>- 训练成本高，需大量仿真数据<br>- 黑盒性强，解释性差 |
| **AP Method** | - 依赖最近一次任务的能耗稳定性（假设短期不变）<br>- 若任务模式突变（如突然大包发送），可能误判 |
| **AsTAR** | - 无法充分利用瞬时高能量窗口（响应慢）<br>- 吞吐量偏低 |
| **All Methods** | - 仿真结果尚未在真实硬件上全面验证<br>- 未考虑多任务并发场景 |

---

### 未来工作方向
1. **真实硬件部署验证**：
   - 测量 RL/AP 在真实 MCU 上的 energy/memory 开销是否符合预期。

2. **混合策略（Hybrid Policies）**：
   - 白天使用 AP 或 RL 最大化吞吐；
   - 夜间切换至 AsTAR 模式保障基本通信。

3. **扩展至多任务与异构负载**：
   - 支持不同类型任务（传感、计算、通信）的优先级调度。

4. **在线学习与持续适应**：
   - 让 RL 或 AP 能够在线更新模型以应对长期环境漂移。

5. **集成到操作系统或 RTOS 层**：
   - 实现通用化的 batteryless-aware task scheduler 框架。

---

> 🔚 **总结一句话**：  
> 在 batteryless IoT 中，**没有银弹式的调度策略**；应根据**硬件配置**（尤其是电容大小）和**应用需求**（吞吐 vs 可靠 vs 连续性）选择最合适的方法——**小电容用动态，大电容用静态；要吞吐选 AP，要节奏选 AsTAR，要灵活可调用 RL**。

</details>

---

### 11. [Beyond Trajectory Imitation: Strategy-Guided Policy Optimization for LLM Reasoning](https://arxiv.org/abs/2606.24064)

**Authors**: Tianyuan Shi, Canbin Huang, Bei Li, Xin Chen, Xiaojun Quan, Jingang Wang, Qifan Wang  
**Category**: cs.AI  
**Published**: 2026-06-24  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.24064v1  

#### Abstract
Distilling reasoning capabilities from strong to weak language models typically involves imitating specific solution trajectories, effectively transferring what to answer rather than how to reason. This trajectory-level imitation encourages memorization of instance-specific steps rather than acquisi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Beyond Trajectory Imitation: Strategy-Guided Policy Optimization for LLM Reasoning —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前将强语言模型（LLM）的推理能力迁移到弱模型的方法（如 SFT 或混合策略优化）普遍依赖于**实例级的轨迹模仿**（trajectory imitation），即让学生模型直接复制专家在特定问题上的完整推理路径。这种做法存在以下缺陷：
- 鼓励**记忆化**（memorization）而非泛化；
- 学习的是“**该写什么**”（what to answer），而不是“**如何思考**”（how to reason）；
- 对新问题或变体问题的**泛化能力差**。

### 🚀 提出的新方法：SGPO（Strategy-Guided Policy Optimization）
作者提出 **SGPO**，一种从“轨迹模仿”转向“**策略蒸馏**”（strategy distillation）的新框架，其核心思想是：
- 不再模仿具体的推理步骤序列，而是从强模型的回答中提取**可复用的策略描述**（strategy description）；
- 利用这些策略引导学生模型生成“有指导”的推理路径，并通过比较“自主”与“策略引导”下的行为差异，将**策略带来的分布变化**蒸馏回无指导策略中。

#### 策略描述的结构（structured strategy description）：
1. **Problem type**：问题类型识别（如“二次方程求解”）
2. **Strategy**：使用的原理/定理/技巧（如“配方法”）
3. **Procedural steps**：高层执行步骤（不包含具体计算或答案）

### 🔍 相比现有方法的优势
| 维度 | 传统方法（SFT/HPT/LUFFY） | SGPO |
|------|---------------------------|------|
| 蒸馏单位 | 实例级 solution trajectory | 可复用 strategy description |
| 是否依赖外部提示 | 是（推理时需提供专家前缀等） | 否（最终策略内化到 unguided policy） |
| 泛化性 | 易过拟合特定路径 | 更好适应新问题 |
| 多样性保留 | 强制模仿单一路径 → 抑制多样性 | Forward-KL + GRPO 保持探索多样性 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **训练数据**：从 LUFFY 数据集中随机采样的 8.5K 数学问题，参考解答由 DeepSeek-R1 生成。
- **策略提取**：基于 DeepSeek-R1 的完整回答，通过 prompt 提取结构化策略描述（见 Appendix B）。

### 🧪 评估基准（四个数学推理 Benchmark）
| Benchmark | 特点 |
|---------|------|
| **MATH500** | 中等难度，标准数学题集合 |
| **AMC23** | 美国数学竞赛题，较难 |
| **OlympiadBench** | 奥赛级别多模态科学问题，极具挑战性 |
| **AIME24** | 高阶数学竞赛题，极难 |

> 所有测试均报告 `avg@32`（AMC/AIME）或 `avg@4`（MATH/Olympiad），温度 0.6，top-p=0.95，最大生成长度 32K tokens。

### ⚙️ 实验设置
- **模型家族**：
  - Qwen2.5-{1.5B, 7B}-Instruct
  - Llama-3.2-8B-Instruct
- **训练流程**：
  1. 先进行 SFT warm-up（约 1/3 数据，10 epochs）
  2. 再进行 RL 阶段训练（10 epochs），每轮每个问题采样：
     - $ G_1 = 8 $ 条 autonomous 轨迹（from $\pi(\cdot|q)$）
     - $ G_2 = 4 $ 条 strategy-guided 轨迹（from $\pi(\cdot|q,s)$）
- **优化目标**：
  $$
  \mathcal{L}(q) = \mathcal{L}_{\text{GRPO}} + \alpha(q) \cdot \mathcal{L}_{\text{KD}}
  $$
  其中 $\alpha(q)$ 是自适应权重，控制蒸馏强度。

### 🆚 基线方法对比
| 方法 | 类型 | 说明 |
|------|------|------|
| **SFT** | 监督微调 | 仅在专家轨迹上做 SFT |
| **SFT+GRPO** | 两阶段 | SFT 后接 GRPO 强化学习 |
| **HPT** | 混合策略 | 在 RL 中加入辅助 SFT 损失 |
| **LUFFY** | 前缀引导 | 将专家轨迹作为 off-policy rollout 输入 GRPO |
| **SGPO（本文）** | 策略蒸馏 | 蒸馏策略引起的分布偏移，非轨迹模仿 |

---

## 3. 主要实验结果和性能指标

### 📊 主要性能结果（平均得分）

| Model | MATH500 | AMC23 | Olympiad | AIME24 | **Average** |
|-------|--------|--------|----------|--------|------------|
| Qwen2.5-1.5B Base | 50.1 | 21.3 | 18.4 | 3.0 | 23.2 |
| SFT | 48.4 | 19.7 | 16.1 | 4.7 | 22.2 |
| SFT+GRPO | 54.1 | 26.2 | 21.7 | 5.5 | 26.9 |
| HPT | 53.8 | 28.5 | 22.6 | 7.7 | 28.2 |
| LUFFY | 54.4 | 27.9 | 23.1 | 8.4 | 28.5 |
| **SGPO** | **57.7** | **29.0** | **23.7** | **9.0** | **29.9** |

| Qwen2.5-7B Base | 75.2 | 43.0 | 38.8 | 11.8 | 42.2 |
| SFT | 76.4 | 42.1 | 40.0 | 12.4 | 42.7 |
| SFT+GRPO | 80.3 | 52.7 | 48.4 | 18.0 | 49.9 |
| HPT | 79.1 | 53.5 | 48.8 | 17.0 | 49.6 |
| LUFFY | 78.0 | 53.0 | 47.6 | 16.4 | 48.8 |
| **SGPO** | **82.7** | **55.9** | **50.0** | **19.7** | **52.1** |

| Llama-3.2-8B Base | 43.7 | 20.3 | 14.5 | 3.0 | 20.4 |
| ... | ... | ... | ... | ... | ... |
| **SGPO** | **52.0** | **25.6** | **22.9** | **10.0** | **27.6** |

✅ **关键结论**：
- SGPO 在所有三个模型上均显著优于最强基线（LUFFY/HPT），**平均提升 2.2 分**；
- 在 Qwen2.5-7B 上实现 **+9.9 分** 的绝对增益（从 42.2 → 52.1）；
- 表现出**互补扩展性**：基础模型越强，SGPO 增益越大。

### 🔬 消融实验结果（Ablation Study on Qwen2.5-7B）

| 设置 | Average |
|------|--------|
| **SGPO（完整）** | **52.1** |
| w/o strategy distillation（退化为 SFT+GRPO） | 49.9 |
| w/o autonomous GRPO | 48.6 |
| w/o all proximal constraints | 45.3 |
| w/o KL clipping | 48.7 |
| w/o target selection | 48.1 |
| w/o adaptive weighting | 50.9 |

📌 发现：
- 移除任何组件都会导致性能下降；
- **proximal constraints 最关键**（-6.8 pts），防止 KL 更新过大导致熵崩溃；
- **autonomous GRPO 至关重要**：缺乏自主探索会导致策略僵化；
- **adaptive weighting 提升效率与上限**：动态调整蒸馏强度更高效。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Forward-KL 蒸馏具有天然选择性**：
   - 自动聚焦于策略关键 token（如“使用求根公式”），忽略格式化等常规 token；
   - 无需显式标注即可区分“决策点”与“表达细节”。

2. **策略蒸馏优于轨迹模仿**：
   - 直接对 guided 轨迹做 SFT（uniform fitting）效果远不如 KL-based distillation（Table 3：52.1 vs 49.3）；
   - 因为后者只吸收“策略引起的变化”，而非整条路径。

3. **训练过程自然过渡**：
   - 早期：$\alpha(q)$ 高 → 依赖策略指导快速起步；
   - 后期：$\alpha(q)$ 下降 → 自主策略主导优化；
   - 整个过程**无需手动调度**（no manual scheduling）。

4. **互补扩展性（complementary scaling）**：
   - 基础模型能力越强，越能有效内化策略知识；
   - 表明：**必须具备一定基础推理能力才能受益于策略蒸馏**。

### ⚠️ 局限性（Limitations）
1. 当前仅验证于**数学推理领域**，尚未拓展至代码生成、逻辑推理、科学问答等；
2. 策略提取依赖强模型（DeepSeek-R1），且质量不可控；
3. 极弱的基础模型可能无法从策略中获益（存在能力阈值）；
4. 训练规模有限（8.5K 问题），更大规模的效果待验证。

### 🔮 未来工作方向
- 开发**自动发现策略**的能力，减少对外部强模型的依赖；
- 探索在其他复杂推理任务中的应用（如 formal logic, program synthesis）；
- 研究**最小必要推理能力门槛**，指导小模型训练策略；
- 扩展到更大规模训练集与更强 base models，验证普适性。

---

> 💡 **一句话总结**：  
> SGPO 成功将知识迁移从“抄作业”升级为“学方法”，通过蒸馏策略引发的概率分布变化，使模型真正学会“如何思考”，并在多个数学基准上实现了对 SFT、RL 和混合策略方法的全面超越。

</details>

---

### 12. [Can Scale Save Us From Plasticity Loss in Large Language Models?](https://arxiv.org/abs/2606.24752)

**Authors**: J. Fernando Hernandez-Garcia, Tom\'as Figliolia, Beren Millidge  
**Category**: cs.AI  
**Published**: 2026-06-24  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.24752v1  

#### Abstract
The loss of plasticity - the ability of a network to learn new information after having already learned older information - is a fundamental challenge in creating artificial neural networks capable of continual learning. Although this phenomenon has been known for decades, it has mostly been studied...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Can Scale Save Us From Plasticity Loss in Large Language Models?*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
该论文研究了**Large Language Models (LLMs)** 在持续学习（continual learning）场景下的 **plasticity loss**（可塑性丧失）问题。  
- **Plasticity loss** 指的是模型在长期训练后，虽然可能保留已有知识（避免 catastrophic forgetting），但逐渐**失去高效学习新信息的能力**。
- 尽管该现象在小规模网络中已被广泛研究，但在现代大规模 Transformer 架构、尤其是自然语言任务中的表现尚不明确。
- 本文首次系统地在**真实、多语言、大规模自然语言流**上验证了 plasticity loss 的存在，并探究了其与模型规模的关系。

### 🚀 提出了什么新方法或新思路
- **提出了一种新的多语言持续学习框架（Multilingual Continual Learning Problem）**：
  - 使用 CulturaX 数据集中的 8 种语言循环训练，每轮语言任务固定为 5B tokens。
  - 在每个周期结束后，用一个**未参与训练的越南语（Vietnamese）探测任务**来评估模型对新数据的学习能力。
  - 探测任务仅用于评估，参数更新被丢弃，确保不污染主训练流程。
- **构建了一个预测 plasticity loss 出现时机的 scaling law**：
  - 首次拟合出 plasticity loss 的 onset（开始恶化的时间点）与模型参数量之间的**幂律关系（power-law scaling）**。

### 🔍 相比现有方法的优势
| 方面 | 本文优势 |
|------|--------|
| **任务真实性** | 使用真实自然语言数据（非合成文本），涵盖多种语言，更贴近实际应用场景 |
| **评估方式** | 引入 held-out 探测任务，能更干净地衡量“学习新知识”的能力，而非记忆旧任务 |
| **规模覆盖广** | 覆盖从 5M 到 314M non-embedding parameters 的多个模型尺寸，支持 scaling analysis |
| **控制变量设计** | 固定 aspect ratio 和 attention-head dimension，减少架构差异干扰，使结果更纯粹反映“scale”影响 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **主训练数据**：来自 [CulturaX](https://aclanthology.org/2024.lrec-main.377) 数据集的 8 种语言子集：
  - 英语、中文（书面）、法语、日语、西班牙语、德语、葡萄牙语、俄语
  - 每语言提供 100B 训练 tokens 和 1B 验证 tokens
- **探测任务数据**：越南语（Vietnamese）
  - 单独构建 20B 训练 tokens 和 1B 验证 tokens
  - **从未出现在主训练序列中**，确保探测任务代表“真正的新知识”
- **Tokenizer**：采用 Qwen3 tokenizer（vocab size = 151,680），具备良好的多语言支持

### ⚙️ 实验设置
- **模型架构**：
  - GPT-style decoder-only Transformer
  - 使用 **pre-layer normalization**, GeLU 激活函数，tie embedding
  - 所有模型保持 **aspect ratio $d_{\text{model}} / L \approx 80$** 和 **attention-head dimension = 64**
- **模型大小**（non-embedding parameters）：
  - 5M, 12M, 27M, 39M, 53M, 83M, 106M, 314M 共 8 个尺度
- **训练配置**：
  - 序列长度：2,048
  - 优化器：AdamW ($\beta_1=0.9, \beta_2=0.95$)，weight decay = 0.1
  - Batch size：0.5M tokens
  - 学习率：通过 grid search 调优后插值得到（5M: 3e-3, 314M: 1e-3）
  - 每个任务训练 9,537 步（≈5B tokens）
  - **每个任务开始时重置 optimizer state**，排除 optimizer stale state 对 plasticity 的影响

### 🎯 评估指标
- **主要指标**：**Validation-loss AUC（Area Under Curve）** on Vietnamese probing task
  - 衡量模型在探测阶段适应新语言的速度和效率
  - AUC 越低 → 学得越快 → plasticity 越强
- **归一化指标**：
  $$
  \Delta \text{AUC}_k = 100 \times \left( \frac{\text{AUC}_k}{\text{AUC}_1} - 1 \right)
  $$
  - 相对于第一轮 cycle 的性能变化百分比
  - 正值表示性能下降（plasticity loss），负值表示仍有正向迁移

### 🔁 基线方法对比
- 本文**没有直接与其他算法进行对比**（如 replay、regularization 等 continual learning 方法）
- 主要对比的是不同 **model scale** 下的行为差异
- 同时设置了两种训练范式作为内部对照：
  1. **Non-stationary continual pretraining**（语言循环切换）
  2. **Stationary multilingual training**（混合所有语言均匀采样）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）Plasticity Loss 的普遍存在
- **所有模型**（5M ~ 314M）最终都表现出 plasticity loss：
  - AUC 曲线先下降（受益于跨语言迁移），后上升（学习能力衰退）
- **小模型更早出现 loss**：
  - 5M 和 12M 模型在第 1–2 个 cycle 后即开始恶化
  - 83M 模型直到第 7 个 cycle 才开始恶化
  - 314M 模型坚持到第 48 个 cycle 以上才明显退化

#### （2）Scaling Law 发现
- 拟合得到 plasticity loss onset（任务实例编号 $T$）与参数量 $P$ 的关系：
  $$
  T = 1.3 \times 10^{-5} \cdot P^{0.8269}
  $$
  - 表明 onset 随参数增长呈 **sublinear（次线性）增长**
  - 即：**增大模型只能延迟 plasticity loss，无法根除**
- **外推预测**：
  - 1B 参数模型预计在约 **1.8T tokens** 后出现 plasticity loss
  - 7B 参数模型需训练至 **~9T tokens** 才会显现

#### （3）Stationary Training 中也存在 Plasticity Loss
- 在静态混合训练中（无 abrupt task switch），**同样观察到 AUC 上升趋势**
- 说明 plasticity loss 并非由“任务突变”引起，而是**长期训练本身的固有问题**

### 📊 与基线方法的对比结果
- 本文未引入外部 baseline 算法
- 但通过对比不同规模模型的表现，得出：
  - 更大模型具有更强的抗 plasticity loss 能力（delayed onset）
  - 但收益递减（diminishing returns），仅靠 scale 不足以解决根本问题

### 🔍 消融实验结果（Correlates Analysis）
作者测量了多个潜在相关因素，发现它们与 plasticity loss 存在关联但**非决定性因果**：

| 指标 | 观察结果 | 是否完全解释 plasticity loss？ |
|------|--------|-------------------------------|
| **Average Parameter Magnitude** | 随训练增加，尤其在小模型中显著 | ❌ 不一致：有时 magnitude 下降但 performance 继续恶化 |
| **Dormant Units (MLP 层)** | 大模型中某些层（如第 8、10 层）出现大量低激活单元 | ❌ 小模型（12M）无明显 dormancy 但仍严重 loss |
| **Collapsed Attention Heads** | 注意力熵降低，集中在少数位置（如 BOS token） | ❌ 变化趋势与 loss 不总一致，部分模型相反 |
| **Lazy Attention Heads** | 注意力分布趋于均匀，信息提取能力下降 | ❌ 出现时间与 loss onset 不匹配 |

> 结论：这些是**伴随现象**，提示网络内部发生病理变化，但尚无单一“smoking gun”。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Plasticity loss 在现代 LLM 中确实存在**：
   - 即使是基于真实自然语言、大规模预训练的 GPT-style Transformer，也会在长期训练后丧失学习新知识的能力。
2. **Scale 可以延缓但不能阻止 plasticity loss**：
   - 更大模型能承受更多训练步数才出现退化
   - 但其缓解效果遵循 **sublinear scaling law**，边际效益递减
3. **Plasticity loss 不依赖 abrupt task changes**：
   - 在 stationary 多语言混合训练中也观察到相同现象
   - 说明这不是 continual learning 特有的 artifact，而是神经网络训练的**普遍性质**
4. **多种网络级病理现象共现**：
   - 包括参数膨胀、MLP 单元 dormancy、注意力头 collapse/lazy
   - 提示未来可通过监控这些指标预警 plasticity loss

### ⚠️ 方法的局限性
- **未提出有效的 mitigation 算法**：
  - 仅诊断问题，未验证任何防御策略（如 weight clipping、reset 等）
- **探测任务单一**：
  - 仅使用越南语作为 probing task，泛化性有待验证
- **未覆盖 decoder-only 以外架构**：
  - 如 T5-style encoder-decoder 或 MoE 架构未测试
- **缺乏机制解释**：
  - 当前仅为 correlation 分析，尚未揭示 plasticity loss 的根本动力学机制

### 🔮 未来工作方向
1. **开发维持 plasticity 的训练算法**：
   - 如结合 weight decay、unit reset（ReDo）、gradient perturbation 等方法
2. **探索 architecture-level 改进**：
   - 设计更 resilient 的 activation functions 或 normalization schemes
3. **建立在线 monitoring 工具**：
   - 利用 dormant unit ratio、attention entropy 等作为 plasticity health indicator
4. **扩展到其他模态与任务**：
   - 验证 vision-language 或 code models 是否也有类似现象
5. **理论建模 plasticity dynamics**：
   - 建立 formal theory 来预测何时会发生 plasticity collapse

---

> 💬 **一句话总结**：  
> **Scale helps, but won’t save us — plasticity loss is a fundamental, scalable challenge in LLM training, requiring architectural and algorithmic innovations beyond mere parameter growth.**

</details>

---

### 13. [cuSBF: A Minimizer-Aware Bloom Filter for Genomic Sequence Data on Modern GPUs](https://arxiv.org/abs/2606.24417)

**Authors**: Tim Dortmann, Markus Vieth, Bertil Schmidt  
**Category**: cs.DC  
**Published**: 2026-06-24  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.24417v1  

#### Abstract
Efficient genomic k-mer indexing depends on approximate membership query (AMQ) structures that must deliver high throughput, low false-positive rates (FPR), and modest memory footprints. The Super Bloom filter (SBF) is attractive for this scenario because minimizer-guided sharding and the Findere sc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*cuSBF: A Minimizer-Aware Bloom Filter for Genomic Sequence Data on Modern GPUs*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统的 **Bloom Filter** 和其变体（如 Blocked Bloom Filter）在处理基因组序列数据时，未能充分利用 **k-mer 序列间的局部性**（locality），尤其是在 GPU 上实现时面临以下挑战：
- **高计算开销**：每个 k-mer 需要独立哈希、滑窗找 minimizer、多级哈希等操作，导致 ALU 瓶颈。
- **寄存器压力大**：中间状态变量多，限制了 SM（Streaming Multiprocessor）上的并发 warp 数量。
- **不规则内存访问**：不同线程可能因 minimizer 不同而分散到不同的 shard，造成原子冲突和冗余加载。

尽管 **Super Bloom Filter (SBF)** 在 CPU 上利用 **minimizer-guided sharding** 和 **Findere scheme** 显著提升了效率和准确性，但其设计不适合 GPU 架构。

### 提出的新方法与创新点
作者提出了 **cuSBF**，一个专为现代 GPU 设计的、面向基因组序列的高性能 SBF 实现，核心创新包括：

- **Sectorized 256-bit Shard Layout**  
  将每个 shard 划分为 4 个 64-bit 字段（words），每个 hash 函数绑定到特定 sector，减少 atomic 操作粒度并提升并行性。

- **Cooperative Shared-Memory Tiling**  
  多个线程协作将长序列片段预加载至 shared memory，避免重复全局内存读取，显著提升访存效率。

- **Warp-Level Shard Sharing**  
  同一 warp 内共享相同 minimizer 的线程协同工作：仅由“leader”加载目标 shard，其余通过 `shfl_sync` 获取，消除冗余 global memory 访问。

- **Segmented Warp Reductions**  
  对连续 targeting 相同 shard 的线程进行分段归约（bitwise-OR），仅由 run head 发起 atomic write，大幅降低原子操作次数。

- **Sequence-Native Design**  
  支持直接处理 FASTA/FASTQ 文件（含 gzip 压缩）、自动拼接记录并过滤跨边界无效 k-mer，无需额外预处理。

### 相比现有方法的优势
- 充分利用了基因组数据中 **super-k-mer 的局部性**，将其转化为 GPU 并行优势。
- 在保持低 **False Positive Rate (FPR)** 的同时，实现了远超现有方案的吞吐量。
- 是首个成功将 SBF 完整机制适配到 GPU 架构的开源、header-only CUDA 库。

---

## 2. 核心实验方法和设置

### 数据集
- **C. elegans 参考基因组**（~97 MiB）：较小、缓存友好的测试集，用于 FPR 分析和参数调优。
- **Human T2T-CHM13 v2.0 参考基因组**（~3 GiB）：大规模、超出缓存的数据集，用于评估持续内存带宽下的表现。

### 实验平台
| 系统 | CPU | GPU | 内存类型 / 带宽 |
|------|-----|-----|----------------|
| **System A** | AMD EPYC 7713P | RTX PRO 6000 (Blackwell) | GDDR7 / 1.8 TB/s |
| **System B** | GH200 Grace Hopper | H100 GPU | HBM3 / 3.4 TB/s |
| **System C** | Intel Xeon W9-3595X | — | DDR5 / 300 GB/s |

> 注：System C 用于运行 CPU 版本的 Super Bloom 参考实现（120 线程）。

### 评估指标
- **Throughput**：插入（insertion）和查询（query）速率（单位：GKmer/s）
- **False Positive Rate (FPR)**：在插入真实 k-mer 后，对 10⁹ 条随机 31-mer 查询的误报数
- **Speed-of-Light Analysis**：分析 SM 利用率、L1/L2/DRAM 利用率，判断瓶颈是计算还是带宽
- **Host-to-Device Transfer Overhead**：评估从主机内存加载 FASTX 文件的实际端到端性能影响

### 基线方法对比
| 方法 | 类型 | 说明 |
|------|------|------|
| **GBBF (cuCollections Blocked Bloom)** | GPU, Append-only | 高性能 GPU Bloom 基线 |
| **Cuckoo-GPU** | GPU, Dynamic | 支持增删查的动态 AMQ |
| **Bulk Two-Choice Filter (TCF)** | GPU, Dynamic | 强调局部性的动态滤波器 |
| **GPU Counting Quotient Filter (GQF)** | GPU, Dynamic | 空间高效的动态 AMQ |
| **CPU Super Bloom** | CPU, SBF | 原始 SBF 的参考实现 |

所有方法统一配置为 **16 bits per k-mer** 的名义内存预算。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（System A, C. elegans）

| 方法 | 插入吞吐量 (GKmer/s) | 查询吞吐量 (GKmer/s) | FPR @ 2³⁹ bits |
|------|------------------------|------------------------|---------------|
| **cuSBF** | **~48.5** | **~112** | **~1.6×10⁻⁷** |
| GBBF (cuCollections) | ~5.3 | ~14.5 | ~2.5×10⁻⁵ |
| CPU Super Bloom | ~0.53 | ~0.48 | **~1.4×10⁻⁸** |
| Cuckoo-GPU | ~0.61 | ~14.0 | ~6.4×10⁻⁶ |

> ✅ **cuSBF 达到最高吞吐量**，且 FPR 显著优于大多数 GPU 基线。

### 与基线方法的对比结果
- **vs. cuCollections GBBF**：
  - 插入快 **9.1×**（C. elegans），**8.2×**（human genome）
  - 查询快 **7.7×**（C. elegans），**7.6×**（human genome）
- **vs. CPU Super Bloom**：
  - 插入快 **92×**（C. elegans），**59×**（human）
  - 查询快 **234×**（C. elegans），**165×**（human）
- **vs. 动态 GPU AMQs**：
  - 插入比 Cuckoo-GPU 快 **79–3400×**
  - 查询比 TCF/GQF 快 **15–50×**

> ⚠️ 在 HBM3 系统上（GH200），cuSBF 相对 GBBF 的优势缩小（插入 2.1×，查询 1.37×），因其更偏向 **compute-bound**，而 GBBF 更易受益于高带宽。

### 参数扫描与消融实验（Parameter Sweep）
在 `k=31` 下对 `(s, m, H)` 进行全面搜索，得出 **Pareto 最优配置**：
> **`(s=28, m=16, H=4)`**

#### 关键发现：
- **H > 4 无益**：增加 hash 数反而提高 FPR 和运行时间（寄存器压力上升）。
- **s ≈ 28 最佳**：过小则 s-mer 重叠过多；过大则重叠不足，削弱 Findere 抑制能力。
- **m = 16 最优**：更短的 minimizer 导致密度上升、super-k-mer 缩短，性能下降。

#### FPR 表现：
- cuSBF (`s=28`) 在 2³⁹ bits 时 FPR 比 GBBF 低 **两个数量级以上**。
- 虽然 CPU Super Bloom FPR 更低（强哈希 + 更大 block），但吞吐差距巨大。

#### Speed-of-Light 分析：
- **cuSBF**：维持 **85% SM 利用率**，即使 filter 超出 L2 缓存仍保持高效。
- **GBBF**：SM 利用率随容量增长急剧下降（<10%），表明严重受制于 global memory 延迟。

---

## 4. 关键结论和发现

### 主要发现
1. **基因组序列局部性是 GPU 加速的关键**  
   cuSBF 成功将 **minimizer-driven locality** 和 **super-k-mer 结构** 转化为 warp-level 协作机制，极大减少了冗余计算和内存访问。

2. **优化方向应从“带宽最大化”转向“本地化计算”**  
   尽管 cuSBF 不是最 bandwidth-scalable 的设计，但它通过 **register-resident masking、shared memory tiling、segmented reduction** 实现了更高的端到端吞吐。

3. **SBF 可以在 GPU 上高效实现**  
   本文首次证明：只要架构感知地重新设计，SBF 不仅能在 GPU 上运行，还能显著超越传统 Blocked Bloom Filter。

4. **FPR 与性能可兼得**  
   在合理参数下（如 `s=28, m=16, H=4`），cuSBF 同时实现 **低 FPR** 和 **高吞吐**，优于通用 GPU Bloom Filter。

### 方法的局限性
- **对大字母表支持有限**：当前基于 64-bit packing，对于 protein 或 triplet 编码等大 alphabet，k 值受限，minimizer 长度被迫减小，影响 FPR。
- **依赖输入驻留 GPU 显存**：若需频繁 H2D 传输，性能会严重下降（见下条）。
- **不支持删除操作**：定位为 append-only 场景，不适合需要动态更新的应用。

### 未来工作方向
- **扩展编码表示**：探索 multi-word 或 SIMD 辅助编码，以支持更大 alphabet 和更长 minimizer。
- **联合优化数据移动**：针对 GH200、DGX Spark 等新型异构系统，进一步优化 host staging 与流水线调度。
- **支持动态操作**：开发支持 insert/delete 的 minimizer-aware Cuckoo 或 quotient 变体。
- **集成至实际流程**：嵌入如 ABySS、BIGSI、Ganon 等工具链中验证端到端收益。

---

> 🔚 **总结一句话**：  
> **cuSBF 通过架构感知的设计，首次将 Super Bloom Filter 的序列局部性优势成功迁移到 GPU，实现了基因组 k-mer 索引在吞吐量与准确率上的双重突破。**

</details>

---

### 14. [Machine Learning Modeling for Real-Time Melt Pool Monitoring in Laser Powder Bed Fusion Additive Manufacturing: A Hybrid Approach](https://arxiv.org/abs/2606.23851)

**Authors**: Inioluwa Emmanuel, Zhuo Yang, Ho Yeung, Xinyao Zhang  
**Category**: cs.LG  
**Published**: 2026-06-24  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.23851v1  

#### Abstract
This work investigates the implementation of artificial intelligence and machine learning (AI/ML) for real-time monitoring in laser powder bed fusion (LPBF) additive manufacturing. We developed a binary image classification framework for distinguishing normal and abnormal melt pool images using a ba...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Machine Learning Modeling for Real-Time Melt Pool Monitoring in Laser Powder Bed Fusion Additive Manufacturing: A Hybrid Approach*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本研究针对 **Laser Powder Bed Fusion (LPBF)** 增材制造中熔池异常实时监测的挑战，解决以下关键问题：
- **纯深度学习模型**（如 ResNet、EfficientNet）虽然分类精度高，但推理延迟长（毫秒级），难以满足 LPBF 工艺控制所需的 **实时性要求**（通常需 <10ms 决策周期）。
- **传统机器学习方法**（如 Random Forest）计算效率高，但直接处理原始像素特征时表现不佳，缺乏从图像中自动提取高级语义特征的能力。
- 当前研究普遍缺乏对 **部署可行性**（deployability）的系统评估，尤其是推理延迟、CPU/GPU 资源消耗等工业落地关键指标。

### 🚀 提出的新方法与创新思路
提出一种 **Hybrid 深度学习架构**：  
> **“预训练 CNN 特征提取器 + Classical ML 分类器”**  
具体为：使用冻结的 **EfficientNetB0** 作为特征提取器，生成 1280 维的深度特征向量，再输入到 **Random Forest** 分类器进行最终决策。

该方法实现了：
- **表征学习与分类解耦**（decoupling representation learning from classification）
- 利用 ImageNet 预训练模型在小样本数据下仍能提取高质量通用视觉特征
- 利用 Random Forest 在低维空间快速分类、抗过拟合、支持 CPU 推理的优势

### 🔍 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **准确性** | 显著优于所有纯深度学习模型（F1 达 0.9451） |
| **推理速度** | 单图推理仅 **1.15ms**（CPU），比 ResNet50 快 **156倍**，比 EfficientNetB0 快 **74倍** |
| **资源需求** | 不依赖 GPU 部署，内存占用低，适合边缘设备或工厂产线集成 |
| **可解释性** | Random Forest 支持特征重要性分析，有助于理解判别依据 |

---

## 2. 核心实验方法和设置

### 📁 数据集
- **来源**：NIST AMMT 平台（Additive Manufacturing Metrology Testbed）
- **材料**：Nickel superalloy 625
- **图像数量**：共 **1,200 张灰度图像**
  - 正常（normal）：600 张
  - 异常（abnormal）：600 张（如 keyholing、lack of fusion、不稳定形态）
- **原始尺寸**：120×120 pixels → 统一 resize 至 **224×224** 用于模型输入
- **标签方式**：基于专家视觉判断，未通过 CT 或显微镜验证缺陷真实性

### ⚙️ 实验设置
- **数据划分**：
  - Stratified 80/20 train-test split → 960 训练 + 240 测试
  - 训练集中进一步划分 90/10 为训练/验证集（864 / 96）
- **预处理流程**：
  - Resize、Normalization
  - 数据增强（仅训练集）：随机水平翻转、旋转、缩放、亮度/对比度调整（label-preserving）
- **硬件环境**：Google Colab 免费版（T4/K80 GPU），TensorFlow 2.16 + scikit-learn 1.3
- **训练策略**：
  - 深度模型：冻结 backbone，仅训练顶部分类头（Global Average Pooling + Dropout + Dense+Sigmoid）
  - Early Stopping（patience=6），最多 20 epochs

### 📊 评估指标
#### 分类性能指标：
- Accuracy
- Precision
- Recall
- F1 Score（主指标）
- AUC-ROC

#### 部署相关资源指标：
- Training Time（分钟）
- Inference Latency（ms/image）
- Peak CPU/GPU Memory Usage

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（Test Set 结果）

| Model | F1 Score | Accuracy | AUC | Inference Time (ms/img) | Training Time (min) |
|-------|----------|----------|-----|--------------------------|----------------------|
| **RF + EfficientNetB0 (Hybrid)** | **0.9451** | **0.9458** | **0.9904** | **1.15** | ~1.6 |
| EfficientNetB0 (Transfer Learning) | 0.7958 | 0.7583 | 0.939 | 85.47 | 26.0 |
| ResNet50 | 0.7500 | 0.6750 | 0.935 | 179.51 | 62.7 |
| MobileNetV2 | 0.6667 | 0.5000 | 0.418 | 85.43 | — |
| Raw Pixel + Random Forest | 0.7027 | 0.7250 | 0.828 | 0.99 | ~1.0 |

> ✅ **Hybrid 方法在所有性能维度上均取得最优平衡**

### 🔁 与基线方法对比结果
| 对比项 | 结果说明 |
|--------|-----------|
| **vs. 纯深度学习模型** | Hybrid 方法不仅推理速度快 **两个数量级**，且 F1 和 AUC 更高，证明其兼具精度与效率优势 |
| **vs. Raw Random Forest** | 尽管原始像素 RF 推理更快（0.99ms），但由于缺乏有效特征表示能力，F1 下降约 24%，AUC 下降明显（0.828 vs 0.990）→ 表明 **深度特征至关重要** |
| **消融实验隐含结论** | 使用 EfficientNetB0 提取特征后接 RF，显著优于直接端到端训练同一网络 → 说明 **“特征提取+轻量分类” 架构更优**于全网络微调（尤其在小数据场景） |

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Hybrid 架构是 LPBF 实时监控的理想选择**：
   - 在仅有 1,200 张标注图像的小样本条件下，**EfficientNetB0 + Random Forest** 实现了接近完美的分类性能（F1=0.9451, AUC=0.9904）。
   - 同时保持 **亚毫秒级推理速度**（1.15ms），完全满足 LPBF 过程控制的时间敏感性需求。

2. **准确率-速度权衡被成功打破**：
   - 传统认知中“高精度必伴随高延迟”的困境，在 hybrid 范式下得以缓解。
   - 通过将复杂表征学习与高效分类分离，实现了 **accuracy 与 latency 的帕累托前沿突破**。

3. **深度特征 + 经典 ML 是工业 AI 的可行路径**：
   - 在数据受限、算力有限的制造业环境中，hybrid pipeline 比 end-to-end DL 更具实用价值。
   - 支持 CPU 部署降低了系统集成成本，提升了可扩展性。

### ⚠️ 局限性
1. **数据集限制**：
   - 样本量较小（1,200 images），且来自单一材料（Ni625）、单台设备（NIST AMMT）。
   - 类别平衡（50/50）不符合实际生产中“异常稀少”的情况，可能影响模型在真实场景中的泛化能力。
   - 缺乏 post-process defect validation（如 CT 扫描），标签基于 in-situ 视觉判断，存在主观偏差风险。

2. **实验设计局限**：
   - 采用单次 train-test split，未进行 k-fold cross-validation，性能稳定性评估不足。
   - 未引入 temporal modeling（如 LSTM、Transformer），仅分析静态帧，忽略熔池动态演化特性。

3. **方法非原创性**：
   - Hybrid 架构本身已在医学影像等领域广泛应用，本文主要贡献在于 **首次将其系统应用于 LPBF 熔池监测并量化其优越性**，而非提出全新算法。

### 🔮 未来工作方向
1. **扩大数据规模与多样性**：
   - 跨材料（Ti6Al4V, AlSi10Mg）、跨平台、多工艺参数采集数据，提升模型鲁棒性和泛化能力。

2. **引入时序建模机制**：
   - 构建视频级监测系统，利用 RNN、3D CNN 或 Vision Transformer 捕捉熔池动态变化模式。

3. **结合多模态传感器数据**：
   - 融合 thermal imaging、acoustic emission、pyrometry 等信号，构建 multi-sensor fusion 框架以提高检测可靠性。

4. **开展闭环反馈控制实验**：
   - 将 anomaly detection 模块接入 LPBF 控制系统，实现 real-time parameter adjustment（如 laser power modulation），迈向真正的智能制造。

5. **探索其他轻量化 hybrid 架构**：
   - 如 MobileNetV3 + XGBoost / LightGBM，进一步优化边缘部署性能。

---

> 💡 **总结一句话**：  
> 本论文通过严谨实证表明，在 LPBF 实时熔池监测任务中，**“EfficientNetB0 提取特征 + Random Forest 分类” 的 hybrid 方法在精度、速度与部署可行性之间达到了最佳平衡，为工业级 AI-QC 系统提供了极具前景的技术路线**。

</details>

---

### 15. [Natural Identifiers for Privacy and Data Audits in Large Language Models](https://arxiv.org/abs/2606.24408)

**Authors**: Lorenzo Rossi, Bart{\l}omiej Marek, Franziska Boenisch, Adam Dziedzic  
**Category**: cs.LG  
**Published**: 2026-06-24  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.24408v1  

#### Abstract
Assessing the privacy of large language models (LLMs) presents significant challenges. In particular, most existing methods for auditing differential privacy require the insertion of specially crafted canary data during training, making them impractical for auditing already-trained models without co...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Natural Identifiers for Privacy and Data Audits in Large Language Models

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前对大语言模型（LLMs）进行隐私审计面临两大核心挑战：
- **形式化隐私审计（如 DP auditing）依赖训练时插入 canary 数据**，无法用于已训练好的模型，导致后验（post-hoc）审计不可行。
- **数据集推断（Dataset Inference, DI）需要一个与嫌疑数据同分布的私有非成员验证集（held-out set）**，但在现实中难以获取。

这些限制严重阻碍了在真实场景下对 LLM 进行可扩展、无需重训的隐私风险评估。

### 提出的新方法与思路
本文提出 **Natural Identifiers (NIDs)** ——一种自然存在于 LLM 训练数据中的结构化随机字符串——作为解决上述问题的关键。

#### 什么是 NIDs？
NIDs 是遵循特定生成规则的结构化随机字符串，例如：
- 密码学哈希值（MD5, SHA-1, SHA-256）
- 缩短的 URL
- 加密货币钱包地址（如 Ethereum Address）
- Java 序列化 ID（`serialVersionUID`）

它们广泛存在于代码仓库（GitHub）、论坛（StackExchange）等互联网文本中，并随预训练语料被自然地纳入 LLM 的训练数据。

#### 核心创新机制
利用 NIDs 的两个关键特性：
1. **格式公开且可复现**：其生成函数 $W(z)$ 已知，因此可以从同一分布中无限生成新的字符串（称为 Generated Identifiers, GIDs）。
2. **天然存在**：无需人工注入即可在训练数据中找到，支持真正的后验审计。

基于此，作者实现了两种新型审计：
- **后验 DP 审计**：将训练数据中的 NIDs 视为“天然 canary”，用生成的 GIDs 构造非成员候选集，通过排名任务评估 DP 参数 $\epsilon$。
- **免验证集的数据集推断（DI）**：直接从嫌疑数据中提取 NIDs 并生成同分布 GIDs 作为验证集，从而绕过对私有 held-out 数据的需求。

### 相比现有方法的优势
| 方面 | 传统方法 | 本方法（NIDs） |
|------|--------|-------------|
| 是否需重训模型 | 是（如 Steinke et al., 2023） | 否 ✅ |
| 是否需构造验证集 | 是（通常不可得） | 否 ✅（自动生成 GIDs） |
| 可扩展性 | 低（受限于 canary 插入成本） | 高 ✅（NIDs 天然丰富） |
| 实用性 | 仅适用于可控训练环境 | 可用于已有模型（如开源 LLM）✅ |

此外，该方法还支持构建 **task-specific NIDs**（如数学题模板），进一步拓展应用场景。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Pile 数据集**：用于训练 Pythia 系列模型（70m 到 12b 参数）
  - 子集包括：GitHub, StackExchange, ArXiv, PubMedCentral, HackerNews 等
- **Dolma 数据集**：用于训练 OLMo-7B 模型
  - 包含多个子集如 OpenWebMath, RefinedWeb, Algebraic Stack 等
- **GSM8K**：小学数学应用题数据集，用于 task-specific NID 实验

> 注：所有实验均在 **已知训练数据构成的开源模型** 上进行，以确保 ground truth 可验证。

### 实验设置

#### （1）DP Auditing 设置
- 在 Pythia 模型上微调 DP-SGD（$\delta=10^{-4}$，不同 $\epsilon$）
- 使用 Pile 测试集中提取的 197 个 NIDs
- 对每个 NID 生成 $c-1$ 个 GIDs，形成大小为 $c$ 的候选集合 $V_i$
- 使用 Min-K% 和 Loss 等 MIA 方法对候选排序
- 评估指标：估计的 $\epsilon_{\text{lower}}$ 下界（95% CI）

#### （2）Dataset Inference (DI) 设置
- 嫌疑集 $D_{\text{sus}}$：包含 100 个真实 NIDs
- 构造 held-out 集：为每个 NID 生成 127 个 GIDs（共 $c=128$）
- 使用多种 MIA 方法（Loss, Min-K%, ReCALL, Hinge 等）提取 membership signal
- 使用梯度提升树分类器学习特征与成员状态的关系（K-Fold）
- 统计检验：Kolmogorov-Smirnov (KS) 检验判断 NID 排名是否偏离均匀分布
- 决策阈值：$p < 0.01$ 表示拒绝零假设（即该数据曾用于训练）

#### （3）消融实验设计
- 控制变量分析：
  - 不同 NID 类型（MD5 vs SHA-512）
  - 不同数量的 NIDs（50–100）
  - 不同 MIA 强度（加入 CAMIA/SURP）
  - 错误实现的 GID 生成器（大小写错误、长度偏差）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）DP Auditing 结果
- **高基数（high cardinality, $c=32$）显著提升审计精度**
  - 尤其在较大 $\epsilon$（如 $\epsilon=100$）时，相比 $c=2$（Steinke 方法），样本复杂度降低数倍
  - 图 3 显示，在相同样本量下，$c=32$ 能更准确逼近理论 $\epsilon$
- **更低的估计方差**
  - 表 7 显示跨 4 个随机种子的标准差很小，说明结果稳定

#### （2）Dataset Inference 结果（表 2 & 表 3）
| 模型 | 嫌疑集类型 | p-value（训练子集） | p-value（测试子集） |
|------|-----------|---------------------|--------------------|
| Pythia 12B | GitHub, ArXiv 等 | ≈ 0.0001（全部 < 0.01） | > 0.06（全部 > 0.01） |
| OLMo-7B | OpenWebMath, RefinedWeb 等 | ≈ 0.0001–0.0003 | 0.8961 |

✅ **所有训练子集均被正确识别（真阳性）**  
✅ **所有测试子集均未被误判（无假阳性）**

> 即：**p < 0.01 ⇒ 曾用于训练；p > 0.01 ⇒ 未使用**

#### （3）Task-Specific NIDs 实验（表 4）
在 GSM8K 数学题上构造模板式 NIDs：
| NID 数量 | p-value |
|---------|--------|
| 50      | 8.43×10⁻¹ |
| 100     | 1.60×10⁻⁶ |

➡️ 随着 NID 数量增加，p-value 单调下降，证明统计功效增强。

---

### 与基线方法的对比结果

| 方法 | 是否需重训 | 是否需 held-out set | 执行时间（分钟） | 成员 p-value | 非成员 p-value |
|------|------------|----------------------|------------------|---------------|----------------|
| Maini et al. (2024) | 否 | 是（需 IID 文本） | ~20 | ~10⁻⁴⁶ 至 10⁻¹²² | ~0.06–0.58 |
| Zhao et al. (2025)（合成数据） | 否 | 否 | **~1300–2100** ❌ | ~10⁻³ 至 10⁻⁵ | 1.0 |
| Zhang et al. (2024a)（注入 canary） | 是 ❌ | 否 | ~21 | ~10⁻²³ 至 10⁻³⁰⁰ | ~0.07–0.54 |
| **本文 NID-DI** | **否 ✅** | **否 ✅** | **~21** | **~10⁻¹⁵⁶ 至 10⁻²¹¹** | **~0.38–0.98** |

✅ **NID-DI 在有效性（p-value 更小）和效率（快 100 倍以上）上全面优于 Zhao et al.**

---

### 消融实验结果

#### （1）NID 类型影响（表 19）
| NID 类型 | p-value |
|--------|--------|
| Java Serialization | 4.17×10⁻²¹¹ |
| SHA-512 | 1.67×10⁻¹⁷⁵ |
| SHA-1 / Ethereum | ~10⁻⁸⁸ |
| MD5 | 7.00×10⁻²³ |

➡️ **更长、结构更复杂的 NIDs 泄露更强信号**

#### （2）GID 生成器错误的影响（表 17）
- 若 GID 大小写错误 → 非成员也出现强信号（p=1.16×10⁻⁵⁴），**导致严重假阳性**
- 若长度偏差 ±1 → 影响较小（因 Min-K% 只关注 top-k tokens）

➡️ **强调必须精确匹配原始 NID 分布**

#### （3）MIA 强度的影响（表 18）
引入更强 MIA（如 CAMIA）后：
- p-value 从 3.31×10⁻¹⁵⁶ 降至 <10⁻³⁰⁰
➡️ **框架性能随 MIA 发展持续提升**

#### （4）NID 数量的影响（表 20）
| NID 数量 | p-value |
|--------|--------|
| 50     | 1.01×10⁻⁶⁶ |
| 100    | 3.31×10⁻¹⁵⁶ |

➡️ **p-value 随 NID 数量单调递减，统计功效显著增强**

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **NIDs 是普遍存在的“天然 canary”**：在主流训练语料（Pile, Dolma）中广泛存在，涵盖数十种类型，总数达数万。
2. ✅ **支持真正意义上的后验隐私审计**：无需修改训练流程或重新训练模型，即可评估 DP 保证。
3. ✅ **使 Dataset Inference 实用化**：通过生成同分布 GIDs，彻底摆脱对私有验证集的依赖。
4. ✅ **高基数（large $c$）带来样本复杂度优势**：相比传统的 1-out-of-2 设置，能更快收敛到紧致的 $\epsilon$ 下界。
5. ✅ **无假阳性**：在所有实验中，对于未参与训练的数据，p-value 始终远高于 0.01，表明方法可靠。

### 方法的局限性
- **依赖数据中存在 NIDs**：若目标数据集不含任何结构化随机串（如纯文学文本），则无法应用。
- **要求 GID 生成器精准建模分布**：任何格式偏差（如大小写、长度）可能导致错误结论。
- **目前主要验证于开源模型**：对闭源商业模型的应用仍需信任假设。

### 未来工作方向
- 自动发现新型 NID 类型（如新兴平台的标识符）
- 扩展至多模态模型（如图像中的哈希水印）
- 开发自动化工具包，供监管机构和第三方审计者使用
- 探索防御策略：如何安全过滤 NIDs 而不影响模型能力

---

> 🔍 **一句话总结**：  
> 本文提出的 **Natural Identifiers (NIDs)** 为 LLM 隐私审计提供了首个真正实用、无需重训、无需私有验证集的解决方案，在 DP auditing 和 dataset inference 任务上均展现出卓越性能与鲁棒性。

</details>

---

### 16. [Safe and Generalizable Hierarchical Multi-Agent RL via Constraint Manifold Control](https://arxiv.org/abs/2606.24010)

**Authors**: Zihao Guo, Jianing Zhao, Ling Li, Hao Liang, Giuseppe Loianno, Yali Du  
**Category**: cs.AI  
**Published**: 2026-06-24  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.24010v1  

#### Abstract
Multi-agent systems are widely used in safety-critical applications that require coordinated behavior under strict safety constraints. Existing approaches face a fundamental trade-off: learning-based methods achieve strong empirical performance but lack theoretical safety guarantees, while control-t...

---

### 17. [The Latent Bridge: A Continuous Slow-Fast Channel for Real-Time Game Agents](https://arxiv.org/abs/2606.24470)

**Authors**: Bojie Li, Noah Shi  
**Category**: cs.AI  
**Published**: 2026-06-24  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.24470v1  

#### Abstract
A real-time agent for general computer use - with games as the most demanding case - must act within tens of milliseconds while still planning over seconds. These two regimes sit at opposite ends of the latency-quality tradeoff. A reasoning VLM (Qwen3-VL-8B-Thinking) deliberates effectively but requ...

---

### 18. [ScaleToT: Generalizing Structured LLM Reasoning for Billion-Scale Low-Activity User Modeling](https://arxiv.org/abs/2606.24605)

**Authors**: Tianbao Ma, Chang Xi, Yichuan Zou, Chengen Li, Linxun Chen, Zilong Lu, Yanan Niu, Zhaojie Liu, Han Li, Kun Gai  
**Category**: cs.AI  
**Published**: 2026-06-24  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.24605v1  

#### Abstract
Accurate user modeling often depends on rich interaction histories, which are unavailable for billions of low-activity users. Large Language Models (LLMs) can infer latent user states from static profiles, but this reasoning becomes unreliable when profiles are sparse, and applying an LLM to billion...

---

### 19. [Themis: An explainable AI-enabled framework for Reinforcement Learning with Human Feedback](https://arxiv.org/abs/2606.24622)

**Authors**: Andreas Chouliaras, Luke Connolly, Dimitris Chatzpoulos  
**Category**: cs.AI  
**Published**: 2026-06-24  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.24622v1  

#### Abstract
Training safe Reinforcement Learning (RL) systems is inherently challenging, with no guarantee of avoiding unwanted behaviors. The most effective defenses against this are (i) transparency through explainability and (ii) alignment via human feedback. While both show promising results, no publicly av...

---

### 20. [Matching Tasks to Objectives: Fine-Tuning and Prompt-Tuning Strategies for Encoder-Decoder Pre-trained Language Models](https://arxiv.org/abs/2606.24841)

**Authors**: Ahmad Pouramini, Hesham Faili  
**Category**: cs.AI  
**Published**: 2026-06-24  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.24841v1  

#### Abstract
Prompt-based learning has emerged as a dominant paradigm in natural language processing. This study explores the impact of diverse pre-training objectives on the performance of encoder-decoder pre-trained language models across generation and question answering tasks, with a focus on commonsense kno...

---

### 21. [Ground Then Rank: Revisiting Knowledge-Based VQA with Training-Free Entity Identification](https://arxiv.org/abs/2606.23881)

**Authors**: Qian Ma, Qiong Wu, Zhengyi Zhou, Yao Ma  
**Category**: cs.CL  
**Published**: 2026-06-24  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.23881v1  

#### Abstract
Knowledge-Based Visual Question Answering (KB-VQA) requires grounding visual queries to external knowledge beyond directly observable content in images. While recent multi modal large language models (MLLMs) show strong perceptual abilities, they struggle on KB-VQA tasks requiring groundings from bo...

---

### 22. [Decoherence as Defence and the Magnitude of Noise Regularisation: A Rigorous N -Qubit Theory of Stochastic Quantum Neural Networks for Adversarially Robust Network Intrusion Detection](https://arxiv.org/abs/2606.24219)

**Authors**: Gautier-Edouard Edouard Filardo (CREOGN)  
**Category**: cs.CL  
**Published**: 2026-06-24  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.24219v1  

#### Abstract
Stochastic quantum neural networks (SQNNs) encode neuronal activations as qubits, synaptic topology as entanglement, and neural noise through a Lindblad master equation. A recent conference study applied a ring-entangled SQNN to collaborative intrusion detection and reached three conclusions: ring e...

---

### 23. [Posterior Refinement: Fast Language Generation via Any-Order Flow Maps](https://arxiv.org/abs/2606.24773)

**Authors**: Manan Agarwal, Sheel Shah, Chanhyuk Lee, Jaehoon Yoo, Jerry Huang, Seunghoon Hong, Aditi Raghunathan, Jinwoo Kim, Nicholas M. Boffi  
**Category**: cs.CL  
**Published**: 2026-06-24  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.24773v1  

#### Abstract
Non-autoregressive generation offers a powerful paradigm for iterative refinement, allowing models to recursively critique, erase and regenerate arbitrary subsets of tokens. However, existing non-autoregressive models fail to realize this potential. Masked Diffusion Models (MDMs) suffer from factori...

---

### 24. [SHERLOC: Structured Diagnostic Localization for Code Repair Agents](https://arxiv.org/abs/2606.24820)

**Authors**: Hovhannes Tamoyan, Sean Narenthiran, Erik Arakelyan, Mira Mezini, Boris Ginsburg  
**Category**: cs.CL  
**Published**: 2026-06-24  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.24820v1  

#### Abstract
LLM agents solve repository-level coding tasks through multi-turn tool use, but utilize half their budget on locating faults before editing. Dedicated localization frameworks have emerged, yet are still evaluated as file retrieval rather than actionable diagnosis, producing locations without the dia...

---

### 25. [Semi-asynchronous Federated Learning in Flower: Framework Extension and Performance Assessment](https://arxiv.org/abs/2606.24230)

**Authors**: V\'ictor Hidalgo-Izquierdo, Carmen Carri\'on, Blanca Caminero  
**Category**: cs.DC  
**Published**: 2026-06-24  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.24230v1  

#### Abstract
This paper presents an extension of the Flower federated learning framework to support Semi-Asynchronous Federated Learning. The proposed approach adapts the traditional synchronous paradigm to better handle client heterogeneity and straggler effects. By introducing a semi-asynchronous training stra...

---

### 26. [VeryTrace: Verifying Reasoning Traces through Compilable Formalism and Structured Verification](https://arxiv.org/abs/2606.24124)

**Authors**: Ninghan Zhong, Ahmet Ege Tanriverdi, Kaan Kale, Sriram Vishwanath  
**Category**: cs.AI  
**Published**: 2026-06-24  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2606.24124v1  

#### Abstract
Multi-step reasoning with Chain-of-Thought (CoT) prompting remains fragile: logical errors or hallucinations in early steps silently propagate, producing confident but incorrect conclusions. This paper presents VeryTrace, a zero-shot verification-and-repair framework that formalizes natural-language...

---

### 27. [Towards Federated Long-Tailed Graph Learning: An Energy-Guided Dual Decoupling Approach](https://arxiv.org/abs/2606.24237)

**Authors**: Lianshuai Guo, Zhongzheng Yuan, Xunkai Li, Meixia Qu, Wenyu Wang  
**Category**: cs.AI  
**Published**: 2026-06-24  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2606.24237v1  

#### Abstract
Federated Graph Learning facilitates collaborative graph modeling across distributed clients while preserving data privacy. However, real-world data categories frequently exhibit long-tailed distributions. Such statistical scarcity severely degrades performance in two ways: it biases the global mode...

---

### 28. [LemonHarness Technical Report](https://arxiv.org/abs/2606.24311)

**Authors**: Kailong Ren, Fubo Sun, Jiachen Liu, Liu Yang, Zimo Yin, Jiaying Li, Congli Yin, Ming He, Yu Huo, Jiawei Liu, Zeping Chen, Yubin Huangfu, Ronghua Li, Yixuan Wu, Xing Su, Yanzhi Xu, Likang Wu, Hongke Zhao, Lei Zhang, Xiaohui Geng, Jianping Fan  
**Category**: cs.AI  
**Published**: 2026-06-24  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2606.24311v1  

#### Abstract
As large language model (LLM) agents are applied to longer tasks, they increasingly modify workspace state across multiple rounds of iteration. However, agents typically observe only tool outputs and log fragments, while the actual state changes occur in the file system. Without explicit workspace b...

---

### 29. [Cycle-Consistent Neural Explanation of Formal Verification Certificates](https://arxiv.org/abs/2606.24414)

**Authors**: Andoni Rodriguez, Alberto Pozanco, Daniel Borrajo  
**Category**: cs.AI  
**Published**: 2026-06-24  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2606.24414v1  

#### Abstract
Formal verification produces machine-checkable certificates that attest to the satisfaction or violation of temporal properties, yet these certificates remain opaque to non-specialist stakeholders. We propose a cycle-consistent neural architecture that generates faithful natural language explanation...

---

### 30. [A specialized reasoning large language model for accelerating rare disease diagnosis: a randomized AI physician assistance trial](https://arxiv.org/abs/2606.24510)

**Authors**: Haichao Chen, Songchi Zhou, Zhengyun Zhao, Shikai Hu, Xianghong Jin, Hongwei Ji, Li He, Shuli Li, Yiming Qin, Xin Tan, Runfeng Shi, Yih Chung Tham, Jiaye Zhu, Ye Li, Ye Jin, Longhao Cao, Dawei Li, Honghan Wu, Hongqiu Gu, Guanqiao Li, Tudor Groza, Chunying Li, Dian Zeng, Weihong Yu, Gareth Baynam, Saumya Shekhar Jamuar, Min Shen, Shuyang Zhang, Bin Sheng, Sheng Yu, Tien Yin Wong  
**Category**: cs.AI  
**Published**: 2026-06-24  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2606.24510v1  

#### Abstract
Rare diseases affect millions of individuals worldwide, yet timely diagnosis remains a major public health challenge due to scarcity of specialized clinical expertise. While large language models (LLMs) show promise to support rare disease diagnosis, current models are constrained by insufficient cl...

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

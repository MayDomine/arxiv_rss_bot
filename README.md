# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-07-20 08:33:23 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [QUADS: Stabilizing NVFP4 Reinforcement Learning for MoE via QUantization-error Alignment across Dual Sides](https://arxiv.org/abs/2607.15810)

**Authors**: Zhengyang Zhuge, Hao Yu, Xin Wang, Zheng Li, Yizhong Cao, Dayiheng Liu, Jianwei Zhang  
**Category**: cs.LG  
**Published**: 2026-07-20  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2607.15810v1  

#### Abstract
Rollout generation is a major bottleneck in Reinforcement Learning (RL) for Mixture-of-Experts (MoE) Large Language Models, motivating low-precision rollout acceleration such as FP8. As an emerging low-precision format, NVFP4 combines fine-grained scaling for accuracy preservation with native W4A4 F...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：QUADS: Stabilizing NVFP4 Reinforcement Learning for MoE via QUantization-error Alignment across Dual Sides

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在基于 **Mixture-of-Experts (MoE)** 大模型的 **Reinforcement Learning (RL)** 中，**rollout 生成** 是训练流程中的主要性能瓶颈。为加速 rollout，研究者尝试使用低精度格式如 **FP8** 或更激进的 **NVFP4 (W4A4)**。然而，直接将 **NVFP4** 应用于 MoE RL 会导致训练不稳定，在约 150 步后出现崩溃。

本文通过分析发现，这种不稳定性并非源于权重量化误差，而是由 **activation quantization error（激活量化误差）** 主导——由于 NVFP4 的 E2M1 格式极粗（仅 8 个正值表示级别），在线重计算的激活值在量化时产生巨大且难以对齐的误差，导致 rollout 与 trainer 之间的 log-probability 差异迅速扩大，破坏策略梯度更新。

### 提出了什么新方法或新思路
作者提出 **QUantization-error Alignment across Dual Sides (QUADS)**，一种双侧量化误差对齐框架，分别从训练端和推理端解决该问题：

- **Asymmetric Quantization-Aware Training (Asymmetric QAT)**  
  在 trainer 侧采用 **W4A16 QAT**：仅对权重进行 fake-quantize 到 NVFP4，而保持激活为 BF16。这样可以对齐权重路径（因权重可同步），但避免在训练中引入额外的激活量化噪声。

- **Residual Activation Compensation (RAC)**  
  在 rollout 侧，针对高残差通道进行二次补偿：识别出量化误差大的 activation 通道，对其残差部分再次进行 FP4 量化并修正，从而减少激活误差，同时保留原生的 W4A4 GEMM 高吞吐优势。

### 相比现有方法的优势
| 方法 | 局限性 | QUADS 的改进 |
|------|-------|-------------|
| Naive NVFP4 (W4A4 rollout + BF16 train) | 快速崩溃，log-prob gap 迅速增大 | 完全避免崩溃，恢复 BF16 级精度 |
| FP8 RL | 吞吐较低（约为 BF16 的 2×） | 吞吐比 FP8 提升 ~16%，接近理论极限 |
| Weight-only NVFP4 (如 QeRL) | 使用 FP16 GEMM，未利用 FP4 Tensor Core | 保留原生 W4A4 GEMM，最大化硬件利用率 |
| Symmetric W4A4 QAT | 同时 fake-quantize 权重和激活，反而放大误差 | 异构设计更合理，显著降低 mismatch |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **训练数据**：
  - 数学推理任务：融合 O4-Mini、verified prover 和 DeepMath（去除中文条目）
  - 编程任务：过滤后的多样化代码生成数据集
- **评估基准（held-out benchmarks）**：
  - **AIME 2024**, **AIME 2025**
  - **HMMT 2025**
  - **LiveCodeBench (LCB)** —— Python-only，统一评测服务打分

### 实验设置和评估指标
- **模型配置**：
  - MoE 25B-A2.8B 模型
  - 训练后端：Megatron-LM（TP=2, EP=16）
  - 推理引擎：SGLang + trtllm mha
  - 硬件平台：NVIDIA Blackwell GPU 集群
- **RL 设置**：
  - 算法：Group Relative Policy Optimization (**GRPO**)
  - 每 prompt 采样 16 个 response
  - Batch size: 256（mini-batch 128）
  - 学习率：1e-6，AdamW 优化器
  - 最大长度：prompt ≤ 3072，generation ≤ 3268
  - 总训练步数：260 steps，每 10 步评估一次
- **评估指标**：
  - 主要指标：**pass@1 accuracy**（平均于四个 benchmark）
  - 辅助指标：
    - per-step log-probability gap（`||log π_infer - log π_train||_max`）
    - rollout throughput（tokens/s）

### 基线方法对比
| 配置 | 描述 |
|------|------|
| **BF16 RL** | 全精度 baseline：BF16 rollout + BF16 training |
| **FP8 RL** | 当前主流低精度方案：FP8 rollout + FP8 training |
| **Naive NVFP4 RL** | 不加对齐的 NVFP4 rollout + BF16 training |
| **QUADS (ours)** | 提出的方法：Asymmetric QAT + Residual Activation Compensation |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2 和 Figure 5）

| Configuration | LiveCodeBench | HMMT 2025 | AIME 2024 | AIME 2025 | **Average pass@1** |
|---------------|----------------|------------|------------|------------|---------------------|
| BF16 RL       | 63.05          | 66.04      | 83.54      | 80.00      | **73.15**           |
| FP8 RL        | 65.27          | 63.54      | 82.91      | 79.79      | 72.88               |
| Naive NVFP4 RL| 24.25          | 48.33      | 66.67      | 66.25      | 51.37               |
| **QUADS (ours)** | **64.79**      | **62.92**  | **83.33**  | **80.42**  | **72.86**           |

> ✅ **QUADS 平均 pass@1 达到 72.86%，几乎完全匹配 BF16 基线（73.15%），远超 naive NVFP4 的 51.37%（提升 +21.49 pts）**

### 与基线方法的对比结果
- **相比 BF16 RL**：
  - 精度差距 < 0.3 pts，在所有 benchmark 上基本持平
  - log-prob gap 控制在 ~0.86，接近不可消除的 engine drift 下限（~0.74）
- **相比 FP8 RL**：
  - 精度相当（72.86 vs 72.88），无明显损失
  - **rollout throughput 提升 ~16%**，得益于 W4A4 FP4 Tensor Core 更高效
- **相比 Naive NVFP4 RL**：
  - 完全避免训练崩溃（reward 曲线稳定上升）
  - log-prob gap 从 ~1.3 降至 ~0.86
  - 平均 pass@1 提升 **21.49 个百分点**

### 消融实验结果

#### （1）W4A16 QAT vs. Symmetric W4A4 QAT（Figure 7）
- **Symmetric W4A4 QAT**（训练时也 fake-quantize 激活）导致更大的 layer-wise mismatch
- **Asymmetric W4A16 QAT** 显著减小激活误差，验证其有效性

#### （2）Dual-side 对齐组件消融（Figure 8）
| 配置 | Average pass@1 |
|------|----------------|
| Naive NVFP4 RL | 51.4% |
| W4A16 QAT Only | 71.1% |
| **QUADS (QAT + RAC)** | **72.9%** |

> ➕ **两个模块独立且互补**：QAT 解决权重不对齐，RAC 解决剩余激活误差

#### （3）通道选择比例影响（Table 3）
| Residual Channel Ratio | max log-prob gap |
|------------------------|------------------|
| 12.5%                  | 0.923            |
| 25%                    | 0.901            |
| 50%                    | 0.854            |
| 75%                    | 0.812            |
| 100%                   | 0.759            |

> ⚖️ **trade-off between accuracy and cost**：选择 top-50% 通道即可获得大部分收益，仅增加 ~10% FLOPs 开销

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Activation quantization error 是 NVFP4 RL 不稳定的主因**，而非 weight quantization。
2. **Weight 可以通过对称 QDQ 路径对齐，但 activation 因在线重计算无法可靠对齐**。
3. **Asymmetric QAT + Residual Compensation 构成有效双侧对齐机制**，可在不牺牲 W4A4 吞吐的前提下恢复 BF16 级精度。
4. **QUADS 实现了精度与效率的双赢**：达到 BF16 级 accuracy，同时比 FP8 提供 ~16% 更高的 rollout throughput。

### 方法的局限性
- **额外计算开销**：虽然 RAC 仅增加 ~10% FLOPs，但仍需定制 Triton kernel 才能维持高性能。
- **非自适应通道选择**：当前使用固定比例（top-k%）选择补偿通道，未考虑动态变化的分布。
- **依赖 Blackwell 架构**：NVFP4 和 FP4 Tensor Core 是 Blackwell 特有功能，不具备跨代通用性。

### 未来工作方向
- 设计 **adaptive residual channel selection** 机制，根据实时统计自动调整补偿范围。
- 将 QUADS 原则扩展至其他 RL 算法（如 PPO、DPO）和其他模型架构（dense LLMs、vision-language models）。
- 探索 **quantization-aware communication**，进一步缩小分布式训练中不同节点间的数值差异。

--- 

> 🔚 **总结一句话**：  
> **QUADS 成功实现了在 MoE 模型上稳定运行 NVFP4 W4A4 rollout 的 RL 训练，首次在保持原生 FP4 GEMM 高吞吐的同时，达到了与 BF16 相当的精度水平，为下一代低精度 RL 系统提供了可行路径。**

</details>

---

### 2. [An MLIR-Based Compilation Method for Large Language Models](https://arxiv.org/abs/2607.15865)

**Authors**: Pengchao Hu, Zhibin Xin, Yifan Chen, Yangyang Zhou, Liang Wang  
**Category**: cs.CL  
**Published**: 2026-07-20  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2607.15865v1  

#### Abstract
Large Language Models (LLMs) have become the dominant workload on modern AI accelerators, yet deploying them on specialized hardware still faces two core challenges: how to import a trained model into a compiler-friendly intermediate representation, and how to efficiently schedule the autoregressive...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*An MLIR-Based Compilation Method for Large Language Models*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代AI加速器上部署 **Large Language Models (LLMs)** 面临两大核心挑战：
1. 如何将训练好的模型导入编译器友好的中间表示（Intermediate Representation, IR）；
2. 如何在片上内存受限的情况下高效调度自回归推理循环（autoregressive inference loop），尤其是处理不同阶段（prompt处理 vs. token生成）的计算差异。

传统手工编写算子库的方式成本高、难以适应快速演进的模型架构，亟需一种自动化、可扩展的编译方案。

---

### 🚀 提出的新方法与创新点

#### （1）基于 **MLIR** 的分层编译框架
- 引入两个定制化 **dialect**：
  - **TopOp**：高层图级 dialect，与源框架（如PyTorch/HuggingFace）和目标芯片无关，用于表达模型语义（如 `top.MatMul`, `top.FAttention`, `top.RMSNorm` 等）。
  - **TpuOp**：目标硬件 dialect，携带量化格式、内存布局、layer-group 分组、物理地址等芯片相关属性，直接面向代码生成。
- 构建标准 lowering pipeline：`TopOp → TpuOp → Binary`，实现从通用模型描述到专用硬件指令的自动转换。

#### （2）三阶段静态拆分策略：`Prefill`, `Prefill_kv`, `Decode`
针对自回归推理中 **prompt并行处理** 与 **逐token生成** 的显著差异，提出将每个 Transformer 层静态编译为三个独立变体：
- **Prefill**：并行处理输入 prompt，query 长度等于 prompt 长度，生成初始 KV Cache。
- **Prefill_kv**：支持历史上下文复用，在多轮对话或长文本续写时合并已有 KV Cache 后进行预填充。
- **Decode**：每次仅处理一个新 token，但需访问不断增长的历史 Key/Value 缓存。

该设计避免了动态形状带来的运行时开销，提升硬件利用率和内存调度确定性。

#### （3）模块化编译流程
- 将大模型按层和阶段切分为多个小的 TopOp 模块（每层每个阶段独立编译），支持：
  - 并行编译加速构建过程；
  - 运行时灵活组合调用（如 `Prefill + Decode` 或 `Prefill_kv + Decode`）；
  - 支持长上下文分段处理（如 1次 Prefill + 多次 Prefill_kv）。

---

### ⚖️ 相比现有方法的优势

| 维度 | 本文方法 | 传统方法 |
|------|--------|---------|
| **框架兼容性** | ✅ 框架无关（TopOp抽象） | ❌ 依赖特定框架（如PyTorch tracing） |
| **硬件适配性** | ✅ 可通过 TpuOp 扩展至多种AI加速器 | ❌ 算子需手动重写 |
| **推理效率** | ✅ 三阶段静态编译减少动态开销 | ❌ 动态shape导致频繁recompilation |
| **内存管理** | ✅ 静态KV Cache布局 + 地址复用优化带宽 | ❌ 动态分配增加不确定性 |
| **部署灵活性** | ✅ 支持多种量化格式（GPTQ/AWQ/AutoRound/F32/BF16/INT8）直通编译 | ❌ 通常只支持特定量化路径 |

此外，相比 vLLM 等采用 **paged attention** 的动态管理系统，本方法更适合强调 **静态确定性** 和 **低延迟控制流** 的专用AI芯片场景。

---

## 2. 核心实验方法和设置

> 注：本文侧重于方法论与系统设计，未提供详尽的端到端性能 benchmark 表格，但明确说明了实现平台、支持模型及部署验证情况。

### 🔧 实验平台与工具链
- **编译器项目**：[TPU-MLIR](https://github.com/sophgo/tpu-mlir)
- **部署演示项目**：[LLM-TPU](https://github.com/sophgo/LLM-TPU)
- **目标硬件**：专有 TPU 类 AI 加速器（具体型号未公开）
- **前端支持**：HuggingFace Transformers 模型检查点导入

### 📦 支持的模型系列
- **语言模型**：Qwen, Llama 系列
- **多模态模型**：InternVL, MiniCPM-V
- **架构类型**：decoder-only Transformer、grouped-query attention、MoE 变种等

### 📈 量化与部署形式
支持以下量化模式的“直通编译”（direct-through compilation）：
- FP32, BF16, FP16
- INT8（对称/非对称）
- 权重量化模型：GPTQ, AWQ, AutoRound

### 🎯 评估方式
- **功能正确性**：输出 token 序列与原始模型一致
- **部署可行性**：成功生成可在目标硬件运行的二进制文件
- **性能优化效果**：通过实际部署反馈验证吞吐量与延迟改善（文中以定性分析为主）

### 🆚 基线对比（隐含）
虽然没有显式列出对比表格，但方法设计明显优于：
- 手工编写 kernel 的传统方式
- 单一动态图编译方案（如 TorchScript + 动态shape）
- 不区分 Prefill/Decode 的统一编译策略

---

## 3. 主要实验结果和性能指标

> 由于本文是系统性论文而非纯性能评测论文，实验结果以定性陈述和工程验证为主。

### 📊 关键性能数据（来自文中描述）
- **编译成功案例**：已成功部署 Qwen、Llama 等主流 LLM 到 TPU 硬件。
- **量化支持能力**：支持 GPTQ/AWQ 权重量化模型的直接解析与编译，无需反量化。
- **内存优化**：
  - KV Cache 使用静态地址规划，避免运行时复制；
  - 通过 `top.Concat` 实现 in-place 更新或地址复用，降低带宽消耗。
- **计算效率优化**：
  - **RoPE positional encoding** 在编译期预计算，运行时通过 `top.Gather` 查表获取，消除逐元素计算开销；
  - **Causal mask** 使用固定小尺寸 mask（如 128×128）重复 tile，大幅减少 mask 数据加载带宽需求。

### 🔍 消融实验（间接体现）
尽管未设 formal ablation study，但以下设计选择体现了其有效性验证：
- 是否启用 `Prefill_kv` 影响多轮对话连续性的支持能力；
- 是否使用 `top.Gather` 替代 runtime RoPE 计算，直接影响 decode 阶段计算负载；
- 是否静态划分三阶段，决定是否需要动态编译或 padding。

这些机制已在实际部署中被采纳，表明其带来显著收益。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **MLIR 是构建 LLM 编译器的理想基础设施**：
   - 多级 dialect 设计天然适合从高层语义到底层硬件的渐进式 lowering。
   - TopOp 实现了真正的 **framework-agnostic** 模型表达，适用于多样化的 LLM 架构。

2. **三阶段拆分（Prefill / Prefill_kv / Decode）是高效部署自回归推理的关键**：
   - 显著降低动态调度复杂度；
   - 允许各阶段独立优化 tensor shape、memory layout 和 operator fusion 策略；
   - 支持灵活的上下文管理和长序列分段处理。

3. **静态编译 + 模块化组织 提升整体系统可靠性与性能**：
   - 每个模块可独立编译、测试、部署；
   - 运行时可根据上下文长度选择最匹配的 decode variant（“decode chunk list”）；
   - 更利于芯片级资源调度与功耗控制。

---

### ⚠️ 方法的局限性
1. **最大长度受限于编译时设定**：
   - Prefill 和 Decode 模块通常基于固定最大长度（如 8K）预编译；
   - 超出此范围需分段处理，增加控制逻辑复杂度。

2. **KV Cache 内存规划为静态分配**：
   - 不同于 vLLM 的 paged attention 动态管理，灵活性较低；
   - 对极长上下文（>100K）的支持可能受限于静态内存划分。

3. **依赖特定硬件特性定制 TpuOp**：
   - 当前 TpuOp 面向特定 TPU 架构设计，迁移到其他架构需重新定义 lowering 规则。

---

### 🔮 未来工作方向
1. **扩展至 speculative decoding**：
   - 利用三阶段框架支持草稿模型 + 验证模型的并行解码。

2. **更激进的 cache tiling 策略**：
   - 支持超长上下文（>100K）下的分块 KV Cache 管理。

3. **应用于多模态语言模型（MLLM）**：
   - 将视觉塔（vision tower）输出接入相同的 `prefill/decode` 循环，统一编译流程。

4. **自动 layer-group slicing 与 fusion 策略搜索**：
   - 结合模型大小与芯片资源，自动决策最优的算子融合与内存调度方案。

---

## 总结

本文提出了一套基于 **MLIR** 的 LLM 编译方法，通过 **TopOp/TpuOp 双层 dialect** 和 **三阶段静态拆分（Prefill/Prefill_kv/Decode）**，有效解决了 LLM 在专用AI加速器上的部署难题。该方法已在 **TPU-MLIR** 和 **LLM-TPU** 项目中落地，支持 Qwen、Llama 等主流模型及 GPTQ/AWQ 等量化格式的端到端部署，具备良好的通用性、可扩展性和工程实用性。

</details>

---

### 3. [Every Microsecond Matters: Achieving Near Speed-of-Light Latency in GPU Collectives](https://arxiv.org/abs/2607.16100)

**Authors**: Siyuan Shen, Anton Korzh, John Bachan, Tiancheng Chen, Arnav Goel, Ludwig Schneider, Pouya Kousha, Zhenhao He, Sylvain Jeaugey, Kamil Iskra, Nishank Chandawala, Jeff R. Hammond, Torsten Hoefler  
**Category**: cs.DC  
**Published**: 2026-07-20  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2607.16100v1  

#### Abstract
GPU collective communication is typically optimized for bandwidth, yet many emerging workloads are increasingly limited by latency. Long-context decode-heavy large language model (LLM) inference is a prime example, where serving large models requires multiple GPUs, and many small collectives lie dir...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Every Microsecond Matters: Achieving Near Speed-of-Light Latency in GPU Collectives*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代大规模 **GPU 集群**在运行 **LLM 推理**（尤其是长上下文、解码密集型任务）和传统 **HPC 应用**时，通信开销已成为性能瓶颈。尽管现有框架（如 NCCL）优化了带宽，但对 **小消息集体通信**（collective communication）的 **延迟敏感性不足**。

具体而言：
- 在 **LLM 解码阶段**，频繁的小规模 `AllReduce` 操作直接位于 **token 生成的关键路径上**。
- 即使是微秒级（μs）的通信延迟也会累积，显著影响 **inter-token latency (ITL)** 和服务成本。

### 🔧 提出的新方法与创新思路

#### （1）识别并消除内存屏障（Memory Barrier）开销
- 发现现有低延迟 `AllReduce` 实现中，**显式内存屏障**（如 `ncclLsaBarrierSession`）引入了超过 **1 μs 的固定开销**，占小消息总延迟的 40% 以上。
- 提出 **Barrier-Free 同步机制**，通过以下技术避免全局同步：
  - **LL Protocol**：将数据与标志位打包为 16 字节原子写入，接收方通过检查标志判断数据就绪。
  - **Sentinel-Based Sync**：初始化缓冲区为特殊值（如 `-NaN`），发送方覆盖后，接收方轮询检测变化。
  - **Bidirectional Communication + Double Buffering**：利用双向数据交换隐式实现信用控制，避免迭代间缓冲区覆写，无需插入 barrier。

#### （2）提出新型 AllReduce 算法：**Two-Shot LL128 Atomic**
- 利用 **NVLink 的缓存行级原子加法**（128-byte atomic add），在目标缓冲区直接累加。
- 当缓存行首元素等于 GPU 数量时，表示所有参与方已贡献，数据就绪。
- 优势：
  - 减少中间缓冲区占用。
  - 仅需约 **3% 的额外带宽开销**（FP32）或 **1.5%**（FP16/BF16）。
  - 更适合中等消息规模和高 GPU 数场景。

#### （3）设计低延迟 API：`ncclLLBuffer`
- 构建于 NCCL 的 **device-side API** 之上，封装 LL、Sentinel、Double Buffering 等机制。
- 提供统一接口用于快速构建自定义低延迟 collective 内核。
- 支持灵活配置（如 `roundRobinFactor` 控制子缓冲区数量）、编译期展开（unrolling）以提升轮询效率。

#### （4）集成至 NCCL 并开放内核选择
- 实现多种新内核并集成到 NCCL 中，可通过环境变量选择：
  - `AllReduce LLBuffer`（one-shot）
  - `AllReduce LLBuffer Twoshot`
  - `AllReduce LL128 Atomic`

---

## 2. 核心实验方法和设置

### 🧪 实验平台
- **硬件**：
  - **GB200 NVL72 系统**：单节点 4 块 Blackwell GPU，72 GPU 共享一个 NVLink 域（130TB/s 带宽）。
  - **Alps 超算系统**：基于 **GH200 Grace Hopper Superchip**，每节点 4 GPU，用于 HPC 测试。
- **软件栈**：
  - NVIDIA vLLM 容器 (`v26.02`)：Ubuntu 24.04, CUDA 13.1, vLLM 0.15.1, PyTorch 2.11, OpenMPI 4.1.9。

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **AllReduce Latency** | 微基准测试中的端到端通信延迟（μs） |
| **Inter-Token Latency (ITL)** | LLM 推理中生成每个 token 的平均耗时（ms/token） |
| **Throughput** | 每秒生成的输出 token 数量（tokens/s） |
| **Cost per 1M Tokens** | 基于 CoreWeave 定价估算的服务成本（美元/百万 token） |
| **Speedup** | 相对于基线的性能提升倍数 |

### 🆚 对比基线方法
| 方法 | 来源 | 特点 |
|------|------|------|
| **NCCL Ring / Tree** | NCCL Legacy Kernels | 传统环形/树形算法，延迟较高 |
| **NCCL AGxLL / RSxLD-AGxST** | NCCL Symmetric Kernels | 最新的 one-shot/two-shot 对称内存内核 |
| **NVSHMEM One-shot** | NVSHMEM v3.5.21 | 基于 PGAS 模型的一次性 AllReduce |
| **MSCCL++ One-shot/Two-shot** | MSCCL++ v0.8.0 | 针对 AI 推理优化的低延迟 collectives |
| **vLLM Custom AllReduce** | vLLM v0.15.1 | 自定义内核，仅支持单节点 |
| **SoL Lower Bound** | 本文计算 | 基于硬件物理极限的速度-of-light 下界 |

---

## 3. 主要实验结果和性能指标

### 📈 微基准测试结果（Microbenchmarks）

#### （1）接近速度-of-light（SoL）下界
- 在 **GB200 上测算 SoL 下界为 1.404 μs**（基于 L2 RTT 和远程存储延迟）。
- 新提出的 `LLBuffer` one-shot 内核在 **2 GPU 场景下达到 1.51 μs**，仅比 SoL 高 **7%**。
- 多数现有实现延迟在 **2–4 μs**，远高于理论极限。

#### （2）AllReduce 延迟对比（128B 小消息）
| 方法 | 延迟 (μs) | 相对 SoL 开销 |
|------|----------|----------------|
| **LLBuffer One-shot (LL)** | ~1.51 | **+7%** |
| NCCL AGxLL | ~1.97 | +40% |
| vLLM Custom AR | ~2.37 | +69% |
| NVSHMEM One-shot | ~2.57 | +83% |

> ✅ 观察：LL 在极小消息更优；Sentinel 在稍大消息因无 flag 开销而胜出。

#### （3）可扩展性表现
- **LL128 Atomic** 在 **GPU 数量增加时表现出更好扩展性**，因其依赖硬件原子操作而非软件 reduction。
- 在 64 GPU 场景下，其相对于传统 two-shot 设计仍保持竞争力。

#### （4）缓冲区大小影响
- **One-shot**：性能随 scratch buffer 增大趋于饱和，过大会增加 NVLink 压力。
- **Two-shot / LL128**：需要足够大的缓冲区以减少迭代次数，**64 MiB** 成为推荐默认值。
- **LL128 最省空间**，因直接在目标地址累加。

---

### 🧠 实际应用案例研究（Case Studies）

#### （1）LLM 推理（vLLM + 多种模型）
| 模型 | 配置 | ITL 改善 | Throughput 提升 | 成本节省（$/1M tokens） |
|------|------|---------|------------------|------------------------|
| Llama-3.1-70B | 4 GPUs | **↓13%** | ↑12.3% | **>$1.0** |
| DeepSeek-V3 | 8 GPUs | ↓11% | ↑14.9% | **>$11.0** |
| Qwen3-235B | 8 GPUs | ↓9% | ↑10.7% | >$10.8 |

> ✅ 最佳配置结合 `LLBuffer`（小消息） + `SymMem`（大消息） + Multicast，实现全范围优化。

#### （2）传统 HPC 工作负载（cuSOLVERMp）
- 测试 `mp__sygvd`（广义对称定阵特征求解器）在 GH200 上的表现。
- 结果显示：
  - **m=32768 矩阵**：性能提升 **+7.0%**
  - **m=65536 矩阵**：性能提升 **+1.5%**
- 表明即使非 AI 应用，低延迟 collectives 也能带来可观收益。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Every μs truly matters**：在 LLM 解码路径上，即使是 **1 μs 的 AllReduce 延迟降低**，也能转化为 **~0.9% 的成本节约**，在万亿 token 规模下意义重大。
2. **Barrier 是主要延迟来源**：现有实现中，**显式内存 barrier 引入的开销高达 1–2 μs**，是突破 SoL 下界的首要障碍。
3. **Barrier-Free 设计可行且高效**：通过 LL、Sentinel、双缓冲等机制，可在保证正确性的前提下完全移除 barrier。
4. **新 API 显著简化开发**：`ncclLLBuffer` 抽象降低了构建高性能 collective 内核的复杂度，促进生态创新。
5. **跨领域受益**：不仅提升 LLM 推理性能，也加速传统 HPC 应用（如 cuSOLVERMp），验证通用价值。

### ⚠️ 局限性
| 限制 | 说明 |
|------|------|
| **依赖 NVLink 硬件特性** | LL128 Atomic 需要缓存行级原子加法，不适用于 PCIe 或其他厂商互联。 |
| **仅支持 Add 操作** | 因依赖交换律，无法用于 `max`, `min`, `mul` 等非交换 reduction。 |
| **数值稳定性略降** | 原子加法顺序不确定，导致浮点求和结果非确定性，误差略高于有序 reduction。 |
| **非确定性行为** | LL128 Atomic 不满足 AllReduce 的 determinism 要求，不适合需要严格一致性的场景。 |
| **当前仅限 Scale-Up 网络** | 聚焦单 NVLink 域内通信，未涉及多节点 scale-out 优化。 |

### 🔮 未来工作方向
1. **建立准确的性能模型**：替代经验驱动的 kernel selection，实现自动最优策略决策。
2. **扩展 API 抽象层级**：提供 warp-level 或 block-level 接口，提升易用性而不牺牲太多性能。
3. **支持更多 reduction 操作**：探索基于硬件 multicast + in-network reduction 的通用低延迟方案。
4. **跨节点低延迟 collectives**：将 barrier-free 思想推广至 hierarchical collectives 的 local phase。
5. **集成至主流框架**：推动 PyTorch、TensorFlow 等默认启用这些低延迟内核。

---

> 💡 **总结一句话**：  
> 本文通过消除内存屏障、设计 barrier-free 同步机制与新型 AllReduce 算法，在 **GB200 上实现了距 SoL 下界仅 7% 的 AllReduce 延迟**，并在 vLLM 和 cuSOLVERMp 中验证了其对 AI 与 HPC 工作负载的显著性能增益，为下一代低延迟 GPU 通信奠定了基础。

</details>

---

### 4. [Learning Faster without Deeper Networks: A*-Inspired Batch Selection for Efficient CNN Training](https://arxiv.org/abs/2607.15745)

**Authors**: Anxhelo Shehu, Enes Stastoli, Arben Cela  
**Category**: cs.LG  
**Published**: 2026-07-20  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2607.15745v1  

#### Abstract
Common practice when training Convolutional Neural Networks (CNNs) is to use randomly shuffled mini-batches. This creates two limitations: slower convergence, and a diminishing learning signal, since many samples are quickly classified as easy during training. We address these inefficiencies with A*...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Learning Faster without Deeper Networks: A*-Inspired Batch Selection for Efficient CNN Training*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前 CNN 训练普遍采用**随机打乱的 mini-batch**策略，存在两个主要缺陷：
- **收敛速度慢**：大量“简单样本”在训练后期提供有限的信息梯度，造成计算资源浪费。
- **学习信号不均衡**：困难或信息量大的样本被低估，而易学样本被重复利用，导致训练效率低下。

### 🚀 提出的新方法：A*-Inspired Batch Selection (A*-BS)
提出一种轻量级、模型无关的**mini-batch 调度策略**，将 batch 选择建模为一个**启发式搜索问题**，灵感来源于经典的 A* 路径搜索算法。

#### 核心思想：
- 将每个 mini-batch 视为搜索空间中的一个“节点”。
- 使用 A*-like 评分函数对 batches 进行排序：
  $$
  f(B_i) = g(B_i) + \lambda h(B_i)
  $$
  - $ h(B_i) $：基于损失的难度度量（如平均交叉熵），衡量 batch 的“信息量”。
  - $ g(B_i) $：重用惩罚项，记录该 batch 已被选中的次数，防止过度重复使用。
  - $ \lambda $：自适应权重，动态平衡探索（exploration）与利用（exploitation）。

### 🔍 相比现有方法的优势
| 对比维度 | 传统方法（如 Curriculum Learning, Importance Sampling） | A*-BS |
|--------|--------------------------------------------------|------|
| **作用粒度** | 多数在 sample-level 操作 | 在 **mini-batch level** 操作，更高效且兼容性强 |
| **难度定义** | 通常依赖预定义规则或外部教师模型（offline） | 完全**内部驱动**，每轮根据当前模型状态动态更新 |
| **采样机制** | 修改梯度权重或删除样本（改变数据分布） | **仅改变 batch 访问顺序**，保留原始 SGD 和 batch 分布 |
| **架构兼容性** | 需修改优化器或网络结构 | **无需改动网络架构或优化器**，可无缝集成到现有 pipeline |

> ✅ **核心优势总结**：A*-BS 是一种**轻量、通用、无侵入性**的训练加速策略，通过智能 batch 排序提升训练效率。

---

## 2. 核心实验方法和设置

### 📚 数据集
- 使用 **MedMNIST-v2** 基准中的全部 **12 个 2D 医疗图像分类子任务**：
  - PathMNIST, DermaMNIST, ChestMNIST, OCTMNIST, PneumoniaMNIST, BreastMNIST, RetinaMNIST, BloodMNIST, TissueMNIST, OrganAMNIST, OrganSMNIST, OrganCMNIST
- 所有图像统一为 **28×28 分辨率**，适合低复杂度模型。
- 图像类型包括 grayscale（单通道）和 RGB（三通道），任务涵盖 binary、multi-class 和 multi-label。

### ⚙️ 实验设置
- **模型架构**：设计了一个极简 CNN（约 **2.25×10⁵ 参数**），远小于 ResNet-18 (~1.12×10⁷) 和 ResNet-50 (~2.36×10⁷)。
  - 结构：`Conv → MaxPool → Conv → MaxPool → Flatten → Dense(128) → Classifier`
- **训练配置**：
  - 优化器：Adam
  - Batch size：128
  - Epochs：100（与 MedMNIST-v2 官方一致）
  - 硬件：Google Colab 上的 **NVIDIA Tesla T4 GPU**
- **评估指标**：
  - **Accuracy (ACC)**
  - **Area Under the ROC Curve (AUC)**

### 🆚 基线方法对比
1. **ResNet-18 和 ResNet-50**（官方报告结果，来自 [4]）
2. **相同轻量 CNN + 随机 batch shuffle**（用于消融实验）

> 注意：A*-BS 与 ResNet 的比较是基于公开基准值，而非重新训练，以确保公平性。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Table I）

| 成绩亮点 | 数据 |
|--------|-----|
| **优于 ResNet-18/50 的任务数量** | **6/12** 个任务在 ACC 和 AUC 上均胜出 |
| **最大相对增益** | 高达 **15%**（如 OCTMNIST、PneumoniaMNIST） |
| **代表性高光表现** | 
| - **OCTMNIST**: A*-BS ACC=89.7% vs ResNet-18=74.3%  
| - **PneumoniaMNIST**: A*-BS ACC=95.8% vs ResNet-18=85.4%  
| - **OrganCMNIST**: A*-BS AUC=0.999 vs ResNet-50=0.992 |

> ❗ 在 ChestMNIST、TissueMNIST、BloodMNIST 上，ResNet 表现更好，说明深度模型在某些任务上仍具优势。

### 🔁 消融实验结果（Table II）——控制变量验证有效性
- **实验设计**：同一轻量 CNN 架构下，仅改变 batch 顺序（随机 vs A*-BS）
- **结果**：
  - 在 **所有 12 个任务** 上，A*-BS 均优于随机 shuffle
  - 平均 ACC 提升显著（如 PathMNIST: +9.0%, BreastMNIST: +7.0%）
  - AUC 同样全面领先

> ✅ 明确证明：性能提升来自于 **A*-BS 策略本身**，而非模型容量。

### ⏱️ 训练时间与计算成本分析（Table III）
- **A*-BS 开销**：相比随机 shuffle，每 epoch 多出约 **22–35% 时间**（主要用于 loss 估计和排序）
- **整体效率远超 ResNet**：
  - 轻量 CNN + A*-BS 总训练时间仅为 ResNet-18 的 **~1/7 到 1/10**
  - 例如 PneumoniaMNIST：CNN+A*-BS ≈ 60s；ResNet-50 ≈ 1045s

> ✅ **结论**：尽管 A*-BS 有轻微开销，但其带来的快速收敛使其在 wall-clock 时间上大幅领先。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **智能 batch 排序可部分补偿模型容量不足**：
   - 即使使用参数量小两个数量级的轻量 CNN，结合 A*-BS 也能在多个任务上超越更深更复杂的 ResNet 模型。
2. **训练动态至关重要**：
   - 收敛速度和最终性能不仅取决于模型结构，也高度依赖于**数据呈现顺序**。
3. **A*-BS 是高效的训练增强机制**：
   - 不修改模型或优化器，即可实现更快收敛和更高精度。
4. **适用于低分辨率医疗影像场景**：
   - 在 MedMNIST 这类资源受限、图像分辨率较低的任务中尤为有效。

### ⚠️ 局限性
1. **非普适性提升**：
   - 在 ChestMNIST 等需要更强表征能力的任务上，A*-BS 无法完全弥补轻量模型的表达力差距。
2. **未进行多随机种子实验**：
   - 当前消融实验仅基于单次运行，缺乏统计显著性检验（作者建议未来做 R > 5 seeds 的 paired test）。
3. **应用范围有限**：
   - 目前仅验证于 2D、低分辨率、单模态医学图像（28×28），尚未扩展至 3D、高分辨率或多模态数据。

### 🔮 未来工作方向
1. 将 A*-BS 应用于 **更深的 CNN 或 Transformer 架构**，探索协同效应。
2. 扩展至 **3D 医学图像（如 MedMNIST-3D）** 和 **多模态融合任务**。
3. 结合 **adaptive optimizers（如 AdamW）** 或 **learning rate scheduling** 进一步优化训练动态。
4. 探索 **理论解释**：为何 A*-like 策略能有效引导 SGD 收敛到更好的极小值？

---

## ✅ 总结一句话
> **通过引入 A*-inspired 的 batch selection 策略，本文证明了“学得聪明”比“堆深网络”更能高效提升 CNN 训练效果，尤其在资源受限的医疗图像分类任务中展现出巨大潜力。**

</details>

---

### 5. [DSWorld: A Data Science World Model for Efficient Autonomous Agents](https://arxiv.org/abs/2607.15901)

**Authors**: Zherui Yang, Fan Liu, Hao Liu  
**Category**: cs.AI  
**Published**: 2026-07-20  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2607.15901v1  

#### Abstract
Despite strong capabilities in data understanding and decision-making, autonomous data science agents still heavily rely on trial-and-error workflows that involve expensive computation. This bottleneck motivates models that can anticipate the effects of data science operations before real execution....

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DSWorld: A Data Science World Model for Efficient Autonomous Agents

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **autonomous data science agents**（如 ML-Master、AIDE）虽然在自动化数据分析任务上表现出色，但严重依赖“试错”式的工作流（trial-and-error workflows），涉及大量昂贵的计算操作（如模型训练、数据处理、评估等）。这导致系统效率低下，**超过86%的执行时间消耗在实际计算而非智能推理上**。

因此，核心问题是：  
> 如何在不进行真实、高成本计算的前提下，让 agent 预判其操作的效果，从而提升训练与推理效率？

---

### 提出了什么新方法或新思路
本文提出两个关键概念与框架：

#### （1）**Data Science World Model**（数据科学世界模型）
- 受视觉领域的 **World Model** 启发（如预测物理世界的未来状态），作者首次将该思想引入数据科学领域。
- 定义为一个能够基于当前 workflow 状态 $ S_t $ 和候选操作 $ A_t $，预测下一环境状态 $ S_{t+1} $ 的模型：
  $$
  S_{t+1} = W(S_t, A_t)
  $$
- 预测内容包括：数据变化、执行反馈、错误信息、性能信号等。

#### （2）**DSWorld 框架**
一个实用化的 Data Science World Model 实现，包含以下四大组件：
- **State Constructor**：将原始环境转换为结构化状态表示（任务、数据统计、日志等）。
- **Router**：判断操作是否需要重计算；轻量操作直接执行，重操作交由模拟器预测。
- **Compiler**：对轻量操作进行真实执行以保证准确性。
- **LLM-based Simulator**：对昂贵操作（如大规模模型训练）进行状态预测，避免真实运行。

此外，还提出了：
- **Reflective World Model Optimization**（反射式世界模型优化）：一种基于 error-aware reflection 的强化学习策略，通过分析预测误差并迭代改进，提升 transition prediction 质量。
- **DSWorld-8K 数据集**：构建了一个包含约 8,000 条高质量 transition 轨迹的数据集，融合真实轨迹与 LLM 合成数据，并附带 Chain-of-Thought 推理解释。

---

### 相比现有方法的优势
| 维度 | 传统方法 | DSWorld |
|------|--------|---------|
| **计算开销** | 依赖真实执行，耗时长 | 大幅减少真实执行，仅关键路径运行 |
| **效率** | 训练/推理慢，受限于硬件 | 加速 RL 训练 ~14×，搜索推理加速 ~3–6× |
| **预测能力** | 无显式建模环境动态 | 显式建模状态转移，支持“预演” |
| **可扩展性** | 扩展性差，难以规模化探索 | 支持高效搜索与策略优化 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
1. **DSWorld-8K**（本文构建）
   - 包含 8,000 条数据科学 workflow 的状态转移样本。
   - 来源：部分来自真实 agent 在 DA-Code、MLE-Dojo 上的任务轨迹；部分通过 LLM（DeepSeek 3.2）合成。
   - 每条样本格式：$ (S, A, S', T) $，其中 $ T $ 是 CoT 推理轨迹。

2. **Predict-before-Execute Benchmark**（Zheng et al., 2026）
   - 用于评估性能排序（Performance Ranking, PR）能力。

3. **自建 Synthetic Evaluation Tasks**
   - 构造了 540 个评估任务，涵盖五类预测能力：
     - ESP（Execution Success Prediction）
     - ETP（Error Type Prediction）
     - ERS（Execution Result Similarity）
     - EKM（Keyword Matching）
     - PP（Performance Prediction）

4. **MLE-Bench Lite**
   - 从 MLE-Bench 中筛选出 21 个任务，用于评估 agent 在下游任务中的表现（medal rate, score）。

5. **DACode Benchmark**（额外验证）
   - 用于进一步验证作为训练环境的有效性。

---

### 实验设置和评估指标

#### 评估场景
- **RQ1**: DSWorld 在 transition prediction 上的表现。
- **RQ2**: 是否能有效加速 agent 的训练与推理。
- **RQ3**: 提出的优化策略是否有效。
- **RQ4**: 数据规模与模型大小的影响。

#### 评估指标
| 任务类型 | 指标 |
|--------|------|
| ESP / ETP / PR / EKM | Accuracy |
| ERS | Embedding Cosine Similarity |
| PP | 1 - RMSE |
| 下游 agent 性能 | Gold/Silver/Bronze medal rate, Median performance, Score |
| 效率 | Training time (min), Inference time (s) |

#### 基线方法对比
分为两类：
1. **零样本 LLM 基线**（Zero-shot）
   - Llama-3.1-8B, Qwen3-8B, DeepSeek-3.2, GPT-4o, o4-mini
2. **微调后 LLM 基线**
   - Llama-3.1-8B-sft, Llama-3.1-8B-grpo
   - Qwen3-8B-sft, Qwen3-8B-grpo
3. **其他执行器对比**
   - Compiler（真实执行）
   - DeepSeek 3.2（直接用作执行模拟器）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ Transition Prediction 总体性能（Table 1）
| 方法 | AVG ↑ |
|------|-------|
| 最强零样本 LLM (o4-mini) | 0.576 |
| Qwen3-8B-grpo（SFT+GRPO） | 0.771 |
| **DSWorld (ours)** | **0.781** |

👉 **超越最强 LLM 基线 35.6%**（相对提升），且在几乎所有子任务上达到 SOTA。

#### ✅ Agent 训练加速效果（Table 2 & Figure 3）
| 模拟器 | Medals (Any) | Score | 训练时间 ↓ |
|--------|-------------|--------|------------|
| Compiler（真实执行） | 11.11 | 18.11 | 335 min |
| **DSWorld** | 9.52 | 17.67 | **277 min** |
| DeepSeek 3.2 | 1.59 | 10.86 | 3854 min |

👉 **DSWorld 实现约 14× 的 RL 训练加速**，同时保持接近 Compiler 的性能。

#### ✅ Agent 推理加速效果（Table 3）
| Agent | 执行器 | 推理时间 ↓ | 加速比 |
|-------|--------|-----------|--------|
| AIDE | Compiler | 4102 s | — |
| AIDE | DSWorld | 676 s | **~6×** |
| ML-Master | Compiler | 1421 s | — |
| ML-Master | DSWorld | 371 s | **~3.8×** |

👉 在多个 search-based agent 上实现 **3–6× 推理加速**，性能基本持平。

#### ✅ 消融实验结果（Ablation Study, Table 1）
| 模型 | AVG |
|------|-----|
| Qwen3-8B（原始） | 0.576 |
| Qwen3-8B-sft（仅 SFT） | 0.763 (+37.5%) |
| Qwen3-8B-grpo（+GRPO） | 0.771 (+1.05%) |
| **DSWorld（完整）** | **0.781 (+1.3%)** |

👉 表明：
- SFT 已带来显著提升 → 合成数据有效；
- GRPO 进一步优化 → 反射式 RL 有助于精炼预测；
- 完整框架最优 → 成分协同增效。

---

## 4. 关键结论和发现

### 主要发现
1. **Data Science World Model 是可行且高效的范式**
   - 首次成功将 World Model 引入数据科学流程建模。
   - 能准确预测代码执行结果、错误类型、输出相似度和性能趋势。

2. **DSWorld 显著提升 agent 效率**
   - 在不牺牲性能的前提下，实现 **~14× 训练加速** 和 **~3–6× 推理加速**。
   - 特别适合 search-based 或 RL-based agent，因其需大量采样评估。

3. **混合执行机制（cost-aware routing）至关重要**
   - 轻量操作真实执行保精度，重操作由 LLM 模拟提效率。
   - 平衡了 accuracy 与 efficiency。

4. **合成数据 + CoT 推理可有效支撑世界模型训练**
   - 利用 MMTU 数据集和 LLM 自动生成多样化状态与动作。
   - 加入 CoT 提升模型对 transition logic 的理解。

5. **模型与数据规模均正向影响性能（Figure 4）**
   - 更多训练数据（至 6.4k）和更大 backbone（至 14B）持续带来收益。
   - 展示了良好的 scaling law。

---

### 方法的局限性
1. **未建模外部工具调用**（external tool-call transitions）
   - 当前聚焦于 Python 代码执行，未覆盖 API、CLI 工具等交互。

2. **预测质量受限于 LLM 能力**
   - 在复杂 workflow 中可能出现 hallucination 或细节遗漏。

3. **合成数据存在分布偏移风险**
   - 尽管经过验证，但仍可能与真实 agent 行为有 gap，影响泛化。

4. **Router 决策可能出错**
   - 若误判轻量操作为重型，则降低效率；反之则影响准确性。

---

### 未来工作方向
- 扩展至多模态与跨平台工具调用建模。
- 引入更强大的 verification 机制提升合成数据质量。
- 探索 online self-improvement 机制，使世界模型随 agent 使用不断进化。
- 结合 memory 机制，建模长期依赖关系。
- 推广至其他领域（如 AutoML、AutoCV）的世界模型构建。

--- 

> 🔚 **总结一句话**：  
> DSWorld 开创性地提出了 **Data Science World Model** 范式，通过 **结构化状态建模 + 成本感知路由 + LLM 模拟器 + 反思式强化学习优化**，实现了对数据科学 workflow 的高效“预演”，在几乎不损失性能的情况下，将 agent 的训练与推理速度提升了 **一个数量级**，为下一代高效自主 agent 提供了重要基础设施。

</details>

---

### 6. [PagedWeight: Efficient MoE LLM Serving with Dynamic Quality-Aware Weight Quantization](https://arxiv.org/abs/2607.16184)

**Authors**: Yuchen Yang, Yifan Zhao, Anisha Dasgupta, Sasa Misailovic  
**Category**: cs.LG  
**Published**: 2026-07-20  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2607.16184v1  

#### Abstract
Mixture-of-Experts (MoE) is a popular class of large language models (LLMs), offering high efficiency and accuracy. However, in KV-cache-intensive serving scenarios, MoEs often exhibit a tension between the GPU memory requirements of the model weights and the growing KV cache. We propose PagedWeight...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PagedWeight: Efficient MoE LLM Serving with Dynamic Quality-Aware Weight Quantization

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **Mixture-of-Experts (MoE)** 大语言模型的推理服务中，存在严重的 **GPU 内存压力**，主要来自两个方面：
- **模型权重（Model Weights）**：MoE 模型参数量大，权重占用大量 GPU 显存。
- **KV Cache**：随着上下文长度增长，KV Cache 占用持续增加。

传统静态量化（如 GPTQ、AWQ）只能在部署前固定精度，无法动态适应运行时内存变化。而 KV Cache 管理技术（如 vLLM 的 PagedAttention）虽优化了缓存，但未解决权重端的内存瓶颈。

因此，论文旨在解决：  
> 如何在 **KV-cache-intensive 场景下**，动态平衡 MoE 模型权重精度与 KV Cache 大小，在不显著牺牲质量的前提下最大化吞吐量和内存效率？

---

### 提出了什么新方法或新思路
提出 **PagedWeight** —— 一种面向 MoE LLM 的动态、质量感知的权重量化管理系统，其核心思想是：

- 将 MoE 专家的 **Any-Precision (AP) 权重位平面（bit-plane）和查找表（LUT）** 视为“权重页”（weight pages），类似 PagedAttention 对 KV Cache 的分页管理。
- 在运行时根据 **KV Cache 压力** 动态调整某些专家线性块（linear-blocks）的量化精度（bitwidth），实现：
  - **Offload**：降低不重要专家的 bitwidth，释放 GPU 内存给 KV Cache。
  - **Reload**：当内存宽松时恢复高精度。
- 引入 **质量感知运行时规划器（Quality-aware Runtime Planner）**，综合以下三要素选择最优降精度策略：
  1. **离线敏感度分析**（Hessian-based sensitivity）
  2. **在线路由统计**（routing mass，保护高频专家）
  3. **提示残差建模**（prompt residual，输入相关的精度影响修正）

此外，系统支持异步页迁移（asynchronous page movement），避免阻塞推理流程。

---

### 相比现有方法的优势
| 方法 | 局限性 | PagedWeight 的优势 |
|------|--------|------------------|
| **静态量化**（如 MxMoE） | 精度固定，无法响应运行时内存压力 | 支持运行时动态调整，更灵活 |
| **全局动态量化**（如 DP-LLM） | 全层统一调整，粒度粗 | 按 expert linear-block 粒度精细控制 |
| **KV Cache 优化**（如 PagedAttention） | 不处理权重内存 | 与之互补，共同提升整体内存利用率 |
| **APL**（Any-Precision LLM） | 支持动态 bitwidth，但无内存驱动机制 | 明确以 KV Cache 压力为触发条件，形成闭环控制 |

> ✅ **核心优势**：首次将“权重分页 + 动态量化 + 质量感知决策”结合，构建了一个 **runtime memory manager**，主动在 **accuracy、memory、throughput** 之间进行权衡。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **校准与训练**：
  - `C4` 数据集用于离线敏感度分析和 prompt residual 模型训练。
- **评估任务**：
  - **Perplexity**：`Wikitext2`, `C4`
  - **推理能力**：`GSM8K`, `MATH-500`
  - **长上下文理解**：`LongBench` 子任务（Passage Retrieval, NarrativeQA, QMSum）
  - **生成吞吐量测试**：自定义长序列生成（seq_len=2048）

---

### 实验设置和评估指标

#### 评估模型（MoE 架构多样性覆盖）：
| Model | Params (B) | Experts | Top-K |
|-------|------------|---------|-------|
| Qwen1.5-MoE-A2.7B | 14.3 | 60+4 | 4 |
| Mixtral-8×7B-v0.1 | 46.7 | 8 | 2 |
| Gemma-4-26B-A4B | 25.2 | 128+1 | 8 |

#### 硬件平台：
- Qwen：NVIDIA RTX 6000 Ada
- 其他：NVIDIA GH200 Grace Hopper
- 后端框架：vLLM v0.20.1

#### 评估指标：
| 类别 | 指标 |
|------|------|
| **质量** | Perplexity（越低越好）、Accuracy（越高越好）、LongBench Score |
| **效率** | Peak GPU Memory Consumption（GB） |
| **性能** | Throughput（Tokens Per Second, TPS） |
| **综合权衡** | Quality-Memory Tradeoff 曲线 |

---

### 基线方法对比
| Baseline | 类型 | 是否支持 MoE | 是否动态 |
|--------|------|-------------|----------|
| FP16 | 浮点基准 | 是 | 否 |
| APL (uniform) | 统一动态量化 | 否 → 作者实现 MoE 版本 | 是 |
| MxMoE | 静态混合精度 | 是 | 否 |
| DP-LLM | 动态层级精度 | 是 | 是 |

> 注：报告中区分了理论内存（theoretical）与实际内存（real），因部分方法需保留高精度副本导致实际开销远高于理论值。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ **FP16 等效精度下的极致压缩**
- PagedWeight 可达到 **FP16 级别的准确率**，同时实现：
  - **最高 72.0% GPU 内存节省**
  - **最高 1.94× 吞吐提升**

#### ✅ **相同内存预算下的质量领先**
- 在相近内存消耗下，相比其他量化方法：
  - **质量最高提升达 39.3%**
  - 仅付出最多 **4.1% 的吞吐损失**

---

### 与基线方法的对比结果

#### （1）Quality-Memory Tradeoff（图5）
- 在所有三个模型（Qwen、Mixtral、Gemma）和四项任务上，PagedWeight 均显著优于所有 baseline。
- 举例：
  - 在 Qwen 上，PagedWeight 在 **16GB 内存**即可逼近 FP16 的 perplexity 表现，而 APL 和 DP-LLM 需更高内存。
  - MxMoE 虽优于均匀量化，但仍落后于 PagedWeight，因其静态策略无法应对动态负载。

#### （2）Long-Context Performance（表2）
| 方法 | Memory (GB) | Average Score |
|------|-------------|---------------|
| FP16 | 35.25 | 17.0% |
| APL-6bit | 14.97 | 17.0% |
| **PagedWeight-10GB** | **9.86** | **17.0%** |

> 💡 **结论**：PagedWeight 用不到三分之一的内存达到了 FP16 的长文本理解水平。

#### （3）Serving Throughput（表3）
| 方法 | B=1 TPS | B=4 TPS | Mem (GB) |
|------|--------|--------|---------|
| Uniform APL | 134.5 | 429.9 | 7.63 |
| **PagedWeight** | **130.1** | **419.4** | **7.63** |

> 📉 最大吞吐下降仅 **3.3% (B=1)** 和 **4.1% (B=4)**，几乎可忽略。

---

### 消融实验结果（表4）

| 配置 | Wikitext2 PPL | C4 PPL |
|------|----------------|--------|
| **PagedWeight (Full)** | **7.22** | **10.06** |
| w/o routing statistics | 7.26 | 10.13 |
| w/o prompt residual | 7.31 | 10.19 |
| w/o page movement | 7.43 | 10.33 |
| w/o global sensitivity | 7.46 | 10.40 |

> 🔍 **发现**：
- 所有组件均有贡献，其中 **page movement** 和 **prompt residual** 影响最大。
- 移除任何模块都会导致质量下降，验证了系统设计的必要性。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **MoE 权重可以像 KV Cache 一样被“分页”管理**，通过动态量化释放内存空间，有效缓解显存压力。
2. **并非所有 expert linear-block 都同等重要**，利用 routing imbalance 和 prompt-specific sensitivity 可精准识别可安全降精度的目标。
3. **动态量化可在极小代价下换取巨大收益**：高达 72% 内存节省 + 接近 FP16 质量 + 几乎无吞吐损失。
4. **PagedWeight 与 PagedAttention 正交互补**，二者结合可进一步提升系统整体效率。

---

### 方法的局限性
1. **依赖 Any-Precision LLM (APL) 格式**：需要特定的 bit-plane 存储格式和 fused kernel 支持，通用性受限。
2. **离线校准成本**：需在 C4 等数据集上预先计算 Hessian 敏感度和训练 prompt residual 模型。
3. **当前仅适用于 MoE 模型**：尚未扩展到 dense LLM。
4. **硬件依赖较强**：最佳效果依赖高性能 CPU-GPU 互连以隐藏 offload 延迟。

---

### 未来工作方向
1. **扩展至更多量化格式**：支持 AWQ、SpQR 等主流格式，提升兼容性。
2. **探索更高效的 prompt-wise sensitivity 预测方法**：减少对离线校准的依赖。
3. **集成进端侧或边缘设备 MoE 推理系统**：适配资源受限场景。
4. **联合优化 KV Cache 与 Weight Page 分配策略**：构建统一的 memory scheduler。

---

> ✅ **总体评价**：PagedWeight 是一个极具工程洞察力的工作，它将系统级内存管理思想引入模型量化领域，开辟了 “**dynamic, quality-aware, fine-grained weight paging**” 这一新范式，对高效 MoE 推理系统的设计具有重要指导意义。

</details>

---

### 7. [CRAFT: Clustering Rubrics to Diagnose Weak LLM Capabilities and Generate Targeted Fine-Tuning Data](https://arxiv.org/abs/2607.16122)

**Authors**: Vipul Gupta, Zihao Wang, Razvan-Gabriel Dumitru, MohammadHossein Rezaei, Aakash Sabharwal, Yunzhong He  
**Category**: cs.AI  
**Published**: 2026-07-20  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.16122v1  

#### Abstract
Evaluations should do more than measure a models current performance. They should tell us what to fix for the next model iteration and provide a way to generate targeted post training data. Most evaluation pipelines identify weak examples, topics, or categories, but they leave the underlying capabil...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CRAFT: Clustering Rubrics to Diagnose Weak LLM Capabilities and Generate Targeted Fine-Tuning Data

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前的 **LLM 评估** 主要停留在任务或类别层面的性能打分（如“模型在数学上表现差”），但无法解释 *为什么* 模型失败（例如是公式选择错误、计算失误还是单位遗漏）。这种粗粒度诊断难以指导后续的 **post-training 数据生成**。

此外，现有方法（如 EvalTree）虽然构建了能力树，但以 **prompt 为单位** 聚类，忽略了单个 prompt 中可能测试多个细粒度能力的事实。

### 提出的新方法：CRAFT
作者提出 **CRAFT**（Clustering Rubrics for Actionable Fine-Tuning），一种将 rubric-based 评估转化为模型特定弱项诊断并用于生成定向微调数据的方法。其核心思想是：
- 将每个 **prompt-rubric pair** 视为一个独立的 **capability probe**（能力探测器）
- 从 rubric 准则中提取能力描述，并聚类成 **层级化的能力树**（hierarchical capability tree）
- 在树的不同层级上动态识别模型表现最差的节点（即“弱点”）
- 利用这些被选中的弱能力节点来 **生成针对性的 fine-tuning 数据**

### 相比现有方法的优势
| 对比维度 | CRAFT | EvalTree | Random Baseline |
|--------|-------|---------|----------------|
| **诊断单元** | Prompt-Rubric Pair（细粒度） | Whole Prompt（粗粒度） | 无结构 |
| **是否建模能力层次** | 是，显式构建 capability tree | 是 | 否 |
| **是否定位弱点** | 是，动态跨层选择低分节点 | 是，基于 prompt 子树 | 否 |
| **数据生成目标性** | 高，聚焦于实测弱项 | 中，基于弱 prompt 类型 | 低，随机采样 |

> ✅ **优势总结**：CRAFT 通过 rubric-level 分析实现了更精细、更具可操作性的模型诊断，从而生成更有针对性的训练数据。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **诊断数据集（用于构建能力树与诊断）**：
  - **PRBench** 的两个子集：
    - **Finance subset**：629 prompts，10,806 rubric criteria
    - **Legal subset**：532 prompts，9,637 rubric criteria
  - 这些数据仅用于 CRAFT 流程中的能力提取、聚类与评分，**不参与最终评估**

- **评估数据集（held-out benchmarks）**：
  - 共 **13 个独立 benchmark**，与诊断数据完全分离，避免污染
  - **Legal domain (7)**：
    - MMLU Law, MBE（法律知识）
    - CaseHOLD（判例推理）
    - SARA（法规应用）
    - ContractNLI（合同理解）
    - Consumer QA（消费者条款）
    - LegalBench（综合法律任务）
  - **Finance domain (6)**：
    - FinanceBench（财报问答）
    - ConvFinQA（金融对话数值推理）
    - TAT-QA（表格+文本混合推理，使用 F1）
    - FiQA-SA（情感分析）
    - FOMC（货币政策立场分类）
    - MLESG（ESG 主题分类）

### 实验设置
- **目标模型**（4 个开源模型）：
  - Qwen3-4B, Qwen3-8B
  - Gemma-3-4B
  - Llama-3.1-8B-Instruct
- **流程统一控制变量**：
  - 所有方法共享相同的：
    - Synthetic data generation 方式（teacher model + prompt conditioning）
    - Fine-tuning 设置（supervised fine-tuning，1k 示例预算）
    - Evaluation protocol（LLM-as-a-judge + 多次 temperature decoding）
- **唯一变量**：fine-tuning 数据的选择策略

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **CRAFT**（本文方法） | 基于 rubric criterion 提取能力 → 构建能力树 → 动态选择弱节点 → 生成数据 |
| **EvalTree** [38] | 将整个 prompt 聚类成能力树，选择表现差的 prompt cluster 生成数据 |
| **Random** | 不构建树也不诊断弱点，直接从领域池中随机抽样 prompt 生成数据（控制暴露于同域数据的影响） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Tables 1 & 2）

#### 📊 法律领域平均准确率（Table 1）
| Model | Random | EvalTree | **CRAFT** |
|-------|--------|----------|-----------|
| Qwen3-4B | 49.9±2.5 | 51.7±1.4 | **53.0±0.8** ✅ |
| Qwen3-8B | 61.2±0.8 | **62.1±0.4** | 61.5±0.5 |
| Gemma-3-4B | 51.4±0.2 | 50.7±0.2 | **51.7±0.2** ✅ |
| Llama-3.1-8B | 46.9±0.7 | 48.2±0.8 | **51.0±0.8** ✅ |

> 🔹 CRAFT 在 **3/4 模型上取得最高 legal 平均成绩**；在 Qwen3-8B 上虽略低于 EvalTree（0.6 pts），但仍在 decoding variance band 内重叠。

#### 💰 金融领域平均准确率（Table 2）
| Model | Random | EvalTree | **CRAFT** |
|-------|--------|----------|-----------|
| Qwen3-4B | 37.6±0.3 | 40.8±1.1 | **46.0±1.4** ✅ |
| Qwen3-8B | 44.1±1.1 | 43.5±0.2 | **45.1±1.9** ✅ |
| Gemma-3-4B | 38.4±0.5 | 38.6±0.0 | **39.8±1.4** ✅ |
| Llama-3.1-8B | 36.6±0.2 | 36.0±0.0 | **42.5±0.8** ✅ |

> ✅ **CRAFT 在所有四个模型上均取得最佳 finance domain 平均性能**，尤其在 Qwen3-4B 和 Llama-3.1-8B 上领先显著（+5pts 以上）。

### 与基线方法的对比结果
| 比较 | 发现 |
|------|------|
| **CRAFT vs. Random** | 显著优于随机选择，说明 **构建能力树 + 弱点定位** 能有效提升 fine-tuning 效果，不仅仅是“更多同域数据”的作用 |
| **CRAFT vs. EvalTree** | 在 finance 领域全面胜出，在 legal 领域多数情况下更好，证明 **rubric-level 单元比 prompt-level 更具诊断价值** |
| **总体胜率** | 在 8 个 model-domain 组合中，CRAFT 在 **7 个组合中达到最强或等效最强性能** |

### 消融实验结果（Table 3）：动态选择 vs 固定层级

| Model | Domain | Fixed-L4 | Fixed-L5 | **Dynamic** |
|-------|--------|----------|----------|-------------|
| Qwen3-4B | Legal | 47.8 | 49.0 | **50.4** ✅ |
| Qwen3-4B | Finance | 43.6 | 42.6 | **47.1** ✅ |
| Llama-3.1-8B | Legal | 43.6 | 40.1 | **50.3** ✅ |
| Llama-3.1-8B | Finance | 42.3 | 42.9 | **43.1** ✅ |

> 🔍 **发现**：固定层级（L4 或 L5）的表现不稳定——有时 L4 更好，有时 L5 更好。而 **dynamic selection 始终最优**，验证了其设计合理性：不同模型的弱点出现在不同抽象层级，应动态适配。

---

## 4. 关键结论和发现

### 主要发现
1. **评估应超越打分，提供“可行动的诊断”**  
   CRAFT 成功将 rubric-based 评估从“报告得分”升级为“指出具体能力缺陷”，并直接驱动 fine-tuning 数据生成。

2. **rubric criterion 是比 prompt 更优的诊断单元**  
   将每个 criterion 视为 capability probe，能捕捉到 prompt-level 方法忽略的细粒度失败模式。

3. **动态跨层选择机制至关重要**  
   模型弱点分布在能力树的不同深度，固定层级会误判（过细或过粗），动态选择能精准定位“信号最强”的失败点。

4. **该框架具有实际工程价值**  
   在相同数据量、训练配置下，CRAFT 生成的数据带来更优下游性能，表明其可用于真实场景中的迭代优化闭环。

### 方法的局限性
- **依赖高质量 rubric 数据**：需要大量专家编写的 rubrics，限制了在非专业领域的应用。
- **聚类与描述依赖 LLM**：能力描述、聚类标签均由 LLM 生成，存在主观性和不一致性风险。
- **合成数据质量受限于 teacher model**：若 teacher model 本身不具备某些能力，则生成的数据可能无效。
- **未探索多轮反馈循环**：当前为单次诊断→训练流程，尚未验证多次迭代的效果。

### 未来工作方向
- 探索自动或半自动 rubric 生成技术，降低人工成本
- 将 CRAFT 应用于其他领域（如医疗、教育）
- 结合 RLHF 或 DPO，利用诊断结果优化对齐过程
- 构建端到端的“诊断-生成-训练-再评估”自动化 pipeline
- 研究如何融合 human-in-the-loop 来修正能力树中的语义偏差

---

> ✅ **一句话总结**：  
> CRAFT 通过将 rubric 准则视为能力探针，构建层级化能力树并动态识别模型弱点，实现了从“知道哪里错”到“知道为什么错”再到“生成针对性修复数据”的完整闭环，在 finance 和 legal 领域显著提升了 fine-tuning 效果。

</details>

---

### 8. [Trainable Spline Representations for Physics-Informed Learning](https://arxiv.org/abs/2607.15751)

**Authors**: Giovanni Canali, Nicola Demo, Gianluigi Rozza  
**Category**: cs.LG  
**Published**: 2026-07-20  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.15751v1  

#### Abstract
This work introduces Physics-Informed Splines (PI-Splines), a structured spline-based architecture for physics-informed learning. Instead of representing the solution of a differential equation with a neural network, PI-Splines directly parametrize the unknown field through a tensor-product B-spline...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Trainable Spline Representations for Physics-Informed Learning**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
传统 **Physics-Informed Neural Networks (PINNs)** 虽然在求解微分方程方面具有灵活性，但在实际应用中面临以下挑战：
- 对网络架构、初始化、优化器等超参数敏感；
- 存在**spectral bias**，难以捕捉高频、多尺度、边界层或尖锐梯度特征；
- 多项损失项（残差、边界条件、初始条件）之间收敛不平衡；
- 参数量大，训练效率低，且缺乏可解释性。

### **提出的新方法：Physics-Informed Splines (PI-Splines)**
本文提出 **PI-Splines**，一种基于 **tensor-product B-spline 扩展** 的物理信息学习框架，其核心思想是：
- 将未知场函数直接表示为 **B-spline 展开形式**，其控制系数（control coefficients）作为可训练参数；
- 保留 PINNs 的残差驱动训练范式，但用结构化样条近似替代神经网络 ansatz。

### **相比现有方法的优势**
| 特性 | PI-Splines | 标准 PINNs / Fourier-feature PINNs / PIKANs |
|------|-----------|---------------------------------------------|
| **参数效率** | 极高，参数量减少达两个数量级（如 169 vs 17025） | 参数量大，依赖过参数化 |
| **局部性与可解释性** | 控制系数具有几何意义，影响局部区域（compact support） | 权重全局作用，缺乏直观解释 |
| **导数计算** | 分析导数（analytical derivatives），无需自动微分（AD） | 依赖 AD，计算开销大且易引入误差 |
| **光滑性控制** | 显式通过样条阶数（spline order）和节点重复度控制 | 隐式由网络深度、激活函数决定 |
| **边界条件处理** | 可强施加（strong imposition）Dirichlet 和 Neumann 条件，移除边界惩罚项 | 通常弱施加（weak enforcement），需平衡损失权重 |

---

## **2. 核心实验方法和设置**

### **使用的基准问题（Benchmark Problems）**
实验涵盖四类典型 PDE 问题，难度递增：
1. **Poisson 问题**：二维椭圆型方程，平滑低频解。
2. **Helmholtz 问题**：各向异性振荡解，测试频率分辨能力。
3. **Exponential 问题**：含指数项的制造解，存在局部尖锐梯度。
4. **Acoustic Wave 问题**：时间依赖波动方程，测试时空联合建模能力。

所有问题均有解析解，用于误差评估。

### **实验设置**
- **实现平台**：基于 **PINA** 框架，在单块 NVIDIA A800 40GB GPU 上运行。
- **统一训练流程**：
  - 优化器：先 Adam（学习率 $10^{-3}$），后 LBFGS 微调；
  - 内部配点（interior collocation points）：5000；
  - 边界/初始配点：固定数量；
  - 重复 5 次独立随机种子实验，报告均值 ± 标准差。
- **评估指标**：
  - **MAE（Mean Absolute Error）**：在独立采样的评估网格上计算；
  - 可训练参数数量（#Parameters）；
  - 总训练时间（Training Time, min）。

### **基线方法对比**
- **Standard PINN**：全连接前馈网络（2 层 × 128 宽度）；
- **Fourier-feature PINN**：相同结构 + Fourier 特征嵌入层（宽度 128）；
- **PIKAN**：Physics-Informed Kolmogorov-Arnold Network，基于可学习样条边的新型网络。

> 所有方法使用相同的 governing equation、loss terms、collocation sets 和优化策略，以隔离架构差异的影响。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见 Table 1）**

| 任务 | 方法 | MAE ↓ | #Params | 训练时间 (min) |
|------|------|--------|---------|----------------|
| **Poisson** | **PI-Spline** | **3.01±1.01×10⁻⁷** | **169** | **2.84±0.16** |
|          | PINN | 2.26±0.47×10⁻⁴ | 17025 | 3.41±0.25 |
|          | PIKAN | 3.31±5.06×10⁻⁴ | 2049 | 8.88±1.07 |
|          | Fourier-PINN | 1.59±0.39×10⁻³ | 16641 | 6.14±0.07 |
| **Helmholtz** | **PI-Spline** | **1.86±0.72×10⁻⁶** | **529** | **3.58±0.36** |
|             | PINN | 2.46±0.19×10⁻³ | 17025 | 4.27±0.04 |
|             | PIKAN | 2.42±0.19×10⁻⁴ | 2049 | 9.98±0.03 |
|             | Fourier-PINN | 1.92±0.39×10⁻³ | 16641 | 6.17±0.08 |
| **Exponential** | **PI-Spline** | **5.67±4.40×10⁻⁶** | **529** | **5.54±0.34** |
|               | PINN | 7.10±1.84×10⁻² | 17025 | 4.76±0.22 |
|               | PIKAN | 2.11±0.82×10⁻² | 3393 | 11.76±0.07 |
|               | Fourier-PINN | 3.72±1.43×10⁻² | 16641 | 6.59±0.06 |
| **Wave** | **PI-Spline** | **3.24±0.68×10⁻³** | **1520** | **24.69±1.58** |
|        | PINN | 3.15±4.25×10⁻² | 17025 | 24.35±3.52 |
|        | PIKAN | 1.45±0.25×10⁻² | 3393 | 53.58±2.03 |
|        | Fourier-PINN | 4.64±2.37×10⁻² | 16641 | 46.87±5.26 |

> ✅ **PI-Splines 在所有任务中均取得最低 MAE，且参数量最少。**

### **与基线方法的对比结果**
- **精度优势显著**：
  - 在 Poisson 问题中，PI-Splines 的误差比最佳神经基线低 **约 3 个数量级**；
  - 在 Exponential 问题中，误差降低超过 **10⁴ 倍**；
  - 即使在最复杂的 Acoustic Wave 问题中，仍保持最优性能。
- **参数效率极高**：
  - 参数量仅为标准 PINN 的 **~1%**（如 169 vs 17025）；
  - 远低于 PIKAN 和 Fourier-PINN。
- **训练速度更快或相当**：
  - 在三个任务中训练最快；
  - 尽管每次前向传播涉及样条基函数计算，但由于参数少、收敛稳定，总体效率更高。

### **消融实验（Sensitivity Analysis）**
- **变量**：样条阶数（spline order）和每维控制点数（#control points）。
- **发现**：
  - **控制点数量是主导因素**：增加分辨率显著降低误差；
  - **样条阶数增益有限**：高阶提升局部逼近能力，但收益随分辨率提高而饱和；
  - 所有问题均呈现“分辨率限制”行为，表明应优先保证足够控制点密度。

---

## **4. 关键结论和发现**

### **主要发现**
1. **PI-Splines 是一种高效、准确、可解释的 PINN 替代方案**：
   - 在多种 PDE 类型下均优于主流神经架构；
   - 特别适合需要**局部结构建模**（如边界层、尖锐梯度）的任务。
2. **结构化表示优于过参数化神经网络**：
   - 准确性不依赖于大量参数，而是来自良好的函数空间设计；
   - **紧凑支持（compact support）** 和 **显式光滑性控制** 是关键优势。
3. **边界条件可强施加**：
   - 利用 clamped B-splines 可精确满足 Dirichlet 和 Neumann 条件；
   - 避免了多目标损失平衡难题。

### **方法的局限性**
- **依赖预设离散化**：
  - 固定节点向量时，逼近空间有限，无法成为通用逼近器；
  - 若分辨率不足，则无法拟合复杂结构。
- **高维扩展成本高**：
  - 张量积结构导致“维度灾难”，高维问题中控制系数数量指数增长；
  - 当前更适合中低维问题（如 ≤3D+time）。
- **实现开销**：
  - 每步需计算样条基及其导数，尤其在密集配点或高阶样条时较慢；
  - 效率受限于样条基评估的实现质量。

### **未来工作方向**
- **自适应细化机制**：
  - 自动调整节点分布或控制点密度（adaptive refinement）；
  - 支持各向异性细化（anisotropic refinement）以应对方向性强的问题。
- **高效实现优化**：
  - 稀疏基矩阵预计算、硬件加速（GPU 向量化）、缓存机制；
  - 探索非张量积样条（如 hierarchical B-splines）以缓解维度灾难。
- **几何灵活性扩展**：
  - 结合 **NURBS** 和 **Isogeometric Analysis (IGA)**，处理复杂几何域；
  - 与 CAD 原生系统集成，用于工业仿真。
- **动态建模范式**：
  - 时间序列预测中采用 autoregressive 形式更新控制系数；
  - 开发时空自适应 discretization 策略。

---

> 🔚 **总结**：  
> **PI-Splines 提供了一种从“黑箱神经网络”转向“白盒结构化表示”的新路径**。它不是要完全取代 PINNs，而是为那些追求**高精度、低参数量、强可解释性和良好边界处理**的应用场景提供了一个强有力的竞争者。该方法展示了将经典数值方法的思想（如样条逼近、强边界施加）与现代机器学习范式结合的巨大潜力。

</details>

---

### 9. [Cost-efficient generative AI summarization for scalable automated essay scoring in educational assessment](https://arxiv.org/abs/2607.15829)

**Authors**: Haowei Hua  
**Category**: cs.CL  
**Published**: 2026-07-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.15829v1  

#### Abstract
Automated essay scoring (AES) enables scalable assessment and timely feedback but remains challenged by transformer input-length limitations, which can cause information loss when processing long essays. This study proposes a generative AI-assisted summarization framework to improve long-form essay ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该研究针对**Automated Essay Scoring (AES)** 中一个关键瓶颈：主流 Transformer 模型（如 BERT、RoBERTa）受限于较短的输入长度（通常为 512 tokens），难以处理长篇学生作文而不进行截断（truncation），从而导致教育相关的重要信息丢失。

### 提出的新方法与思路
提出了一种**生成式 AI 辅助的摘要框架（generative AI-assisted summarization framework）**，其核心思想是：
- 利用 **GPT-5 系列模型**（GPT-5, GPT-5 mini, GPT-5 nano）对原始长文本进行可控长度的摘要生成；
- 将生成的摘要作为下游评分模型的输入，以满足固定长度嵌入模型的要求；
- 同时保留从**原始全文提取的手工语言学特征**（handcrafted linguistic features），形成一种**混合表示（hybrid representation）**，兼顾语义压缩与表层语言信号。

### 相比现有方法的优势
- **相比简单截断（naive truncation）**：避免了直接丢弃文章后半部分可能造成的关键论据或组织结构信息损失。
- **相比长上下文编码器（如 Longformer）**：提供了一种更灵活、可插拔的预处理方案，适用于已基于固定长度嵌入构建的现有 AES 系统，无需重新训练整个模型。
- **相比端到端 LLM 直接评分（direct LLM scoring）**：降低了计算成本和提示敏感性（prompt sensitivity），同时通过融合手工特征增强了解释性和构造效度（construct validity）。
- **强调成本效率（cost-efficiency）**：系统比较不同规模 GPT 模型在性能与 API 成本之间的权衡，为实际部署提供决策依据。

---

## 2. 核心实验方法和设置

### 数据集
- 使用 **ASAP2.0 数据集**（即 Kaggle 上的 *Learning Agency Lab - Automated Essay Scoring 2.0* 竞赛数据集）。
- 包含约 24,000 篇学生议论文，每篇由专家打分为 1–6 分的整数分数（holistic scoring）。
- 高分段作文普遍更长，许多超过 1,300 tokens，显著超出标准 Transformer 的 512-token 限制。

### 实验设置
#### 摘要模块
- 使用三种 GPT-5 变体生成摘要：**GPT-5**, **GPT-5 mini**, **GPT-5 nano**。
- 设定目标输出长度 ≤ 512 tokens；若超限，则采用迭代重摘要策略（每次缩减至前次目标的 80%）直至合规。
- 所有模型使用相同提示词（prompt），确保公平比较。

#### 表示与建模
- **语义嵌入**：使用 **Qwen3-Embedding-4B** 模型将摘要文本编码为固定维度向量。
- **手工特征**：从原始全文提取 22 个手工程征，涵盖五个维度：
  - 表面统计（如字数、句长）
  - 词汇丰富度（TTR, MTLD）
  - 连贯性与话语特征（discourse connectives）
  - 可读性指数（Flesch-Kincaid, Gunning Fog）
  - 错误率（拼写、语法）
- **分类器**：采用 **XGBoost** 和 **LightGBM** 构建梯度提升树模型，融合嵌入向量与手工特征进行最终评分预测。

#### 评估指标
| 类别 | 指标 |
|------|------|
| **下游评分性能** | Quadratic Weighted Kappa (**QWK**) |
| **摘要质量** | ROUGE-1/2/L F1, Semantic Similarity (cosine), Entity Coverage, Keyphrase Overlap, Compression Ratio, Redundancy Score |
| **经济成本** | API 调用费用（$/1M tokens） |

### 基线方法对比
尽管未完整执行所有基线实验，作者明确指出应与以下五类方法对比：
1. **Raw-text truncation**（截断至 512 tokens）
2. **Long-context encoders**（如 Longformer）
3. **仅摘要嵌入模型**（无手工特征）
4. **仅手工特征模型**（无嵌入）
5. **Zero-shot/few-shot LLM scoring**（如 GPT-4、Claude 等直接评分）

> 注：本文聚焦于“在固定混合管道内比较不同 GPT 摘要器”的控制实验，而非全面 SOTA 对比。

---

## 3. 主要实验结果和性能指标

### 下游评分性能（QWK）
| Model | QWK Score |
|-------|-----------|
| **GPT-5 mini** | **0.8435** ✅ |
| GPT-5 | 0.8350 |
| GPT-5 nano | 0.8332 |

- **GPT-5 mini 在 QWK 上表现最佳**，略优于更大更贵的 GPT-5。
- 表明最大模型不一定带来最优评分效果，存在“性价比拐点”。

### 摘要质量综合评估（见 Table 5）
| Metric | GPT-5 | GPT-5 mini | GPT-5 nano |
|--------|--------|------------|-------------|
| **ROUGE-1 F1** | **0.9261** | 0.9137 | 0.9001 |
| **Semantic Similarity** | **0.9843** | 0.9800 | 0.9639 |
| **Entity Coverage** | **0.9425** | 0.9330 | 0.8894 |
| **Compression Ratio** | 0.8933 | 0.8940 | **0.9023** ✅ |

- **GPT-5 整体摘要质量最高**，在几乎所有保留性指标上领先。
- **GPT-5 mini 性能紧随其后**，差距微小。
- **GPT-5 nano 压缩最激进**（最高压缩比），但信息保留能力明显下降，尤其在实体覆盖方面。

### 成本对比（Table 2）
| Model | Input ($/1M) | Output ($/1M) |
|-------|--------------|----------------|
| GPT-5 | $1.25 | $10.00 |
| GPT-5 mini | $0.25 | $2.00 |
| GPT-5 nano | $0.05 | $0.40 |

- **GPT-5 mini 输入/输出成本仅为 GPT-5 的 20%**，而性能几乎持平甚至反超。
- **GPT-5 nano 最便宜**，但性能有所牺牲。

### 关键趋势图分析（Figs. 3–6）
- 所有模型均显示：**随着作文得分等级提高（1→6），摘要质量各项指标（semantic similarity, ROUGE, entity coverage, keyphrase overlap）系统性下降**。
- 原因推测：高分作文更长、更复杂，需更强压缩，易导致信息丢失。
- 小模型（尤其是 GPT-5 nano）在高分段性能衰减更严重。

### 相关性分析（Fig. 7）
- ROUGE-1/2/L 几乎完全正相关（r ≈ 0.99–1.00），说明它们衡量相似维度。
- 内容保留类指标（ROUGE, entity, keyphrase）彼此强相关，且与 compression ratio 正相关。
- **redundancy score 与其他指标弱相关**，表明其反映的是独立的风格效率维度。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **生成式摘要可用于有效缓解 Transformer 输入长度限制**，支持大规模 AES 系统处理长文本。
2. ✅ **GPT-5 mini 是最优选择**：在评分准确性（QWK）、摘要质量和成本之间实现了最佳平衡。
3. ⚠️ **摘要并非中立操作**：对高分、复杂的作文压缩时更容易丢失关键信息，可能导致对高水平写作能力的低估。
4. 💡 **混合表示（hybrid representation）具有价值**：结合摘要语义嵌入与原始文本手工特征，有助于保留被压缩过程遗漏的语言信号。
5. 💬 **模型大小 ≠ 性能最优**：最大的 GPT-5 并未产生最好的评分结果，反而 GPT-5 mini 因更高效的摘要输出获得了更高 QWK。

### 方法的局限性
1. ❌ **缺乏完整基线对比**：未运行 truncation、Longformer、direct LLM scoring 等关键对照实验，无法宣称绝对优势。
2. ❌ **单一数据集与语言**：仅在英文议论文数据集上验证，泛化性未知（如叙事文、多语言、学科特定写作）。
3. ❌ **固定提示设计**：未探索不同 prompt engineering 对摘要质量的影响。
4. ❌ **仅使用一种 embedding model**（Qwen3-Embedding-4B），结果依赖于该模型特性。
5. ❌ **未考虑学术诚信问题**：未讨论 AI 代写背景下如何区分真实学生写作与 LLM 生成内容。

### 未来工作方向
1. 🔄 开展完整的 ablation study，纳入 truncation、Longformer、direct LLM scoring 等基线。
2. 🔁 探索**自适应摘要策略**（adaptive summarization），例如根据原文长度或质量动态调整压缩强度。
3. 📊 在多种数据集、文体、语言环境下验证方法鲁棒性。
4. 🤖 引入**score-aware 或 construct-preserving summarization**，确保高阶写作特征（如论证深度、修辞技巧）在摘要中得以保留。
5. 🌍 推动**低成本、可持续的教育 AI 应用**，关注能源消耗与资源公平获取问题。
6. 🛡️ 加强对**公平性与偏见**的研究，防止摘要过程引入系统性偏差（如对高分学生不利）。

</details>

---

### 10. [BayesPO: Bayesian Prompt Optimization via Parallel-Tempered Gradient-Guided Discrete MCMC](https://arxiv.org/abs/2607.16001)

**Authors**: Junjie Zhou, Zhijian Ou  
**Category**: cs.CL  
**Published**: 2026-07-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.16001v1  

#### Abstract
Prompt optimization adapts large language models (LLMs) without updating model parameters, but many automatic prompt optimizers remain heuristic search procedures over candidate instructions. This paper studies prompt optimization as Bayesian posterior sampling over discrete prompt tokens. We define...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：BayesPO: Bayesian Prompt Optimization via Parallel-Tempered Gradient-Guided Discrete MCMC

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
- **Prompt Optimization** 是一种无需更新 LLM 参数即可适配下游任务的方法，但现有方法多为启发式搜索（如 AutoPrompt、APE），缺乏理论基础。
- 现有方法通常在有限候选集中进行优化，或依赖 LLM 生成新提示，容易陷入局部最优，且未显式建模 prompt 的后验分布。

### 提出了什么新方法或新思路
提出 **BayesPO**（Bayesian Prompt Optimization）框架，将 prompt 优化形式化为 **离散 token 空间上的贝叶斯后验采样问题**：
- 定义 prompt 的能量函数 $ U(p) = -\log P(p) - \sum_i \log P(y_i|p, x_i) $，其中：
  - $ P(p) $：由语言模型提供的 **prompt prior**，鼓励语法流畅性；
  - $ P(y_i|p, x_i) $：任务 likelihood，衡量 prompt 对输入输出示例的解释能力。
- 将优化转化为 **energy-based posterior sampling**，通过 MCMC 进行采样。

### 核心技术创新
1. **Gradient-Guided Discrete MCMC**：
   - 使用 **Gibbs-with-Langevin (GwL)** 提案机制，在离散词表中基于梯度引导生成 token 替换建议。
   - 引入 **Metropolis-Hastings (MH) correction** 保证马尔可夫链收敛到目标分布。
2. **Parallel Tempering (PT)** 集成：
   - 多温度链并行运行，高温度链跨越能量壁垒，低温度链保留高质量状态。
   - 通过 **state swap** 改善全局探索能力，避免陷入局部最优。
3. **适配非权重绑定（non-weight-tied）LLM embeddings**：
   - 区分 input 和 output embedding 梯度来源，使方法适用于现代指令调优模型（如 Qwen2.5）。

### 相比现有方法的优势
| 方法 | 是否显式建模 posterior | 是否保证收敛 | 是否支持全局探索 | 是否利用梯度 |
|------|------------------------|--------------|------------------|---------------|
| AutoPrompt / APE | ❌ 启发式搜索 | ❌ | ❌ | ✅（局部） |
| Reprompting (Gibbs) | ❌ | ❌（无 MH） | ❌ | ❌（LLM proposal） |
| **BayesPO** | ✅（EBM + MCMC） | ✅（MH correction） | ✅（PT） | ✅（embedding gradient） |

> BayesPO 是首个将 prompt 优化建立在 **贝叶斯采样理论框架** 下，并结合 **梯度引导 + MH 校正 + 并行回火** 的完整方案。

---

## 2. 核心实验方法和设置

### 使用的数据集
1. **诊断任务（Diagnostic Tasks）**
   - Classical Chinese Translation：英文 → 文言文风格中文
   - Antonym Generation：中文词语 → 反义词
   - 每个任务使用 **K=4** 个 input-output 示例作为优化集。

2. **诗歌补全任务（Poetry Completion）**
   - 基于李白《静夜思》设计控制实验：
     - 固定前缀：“下面是《静夜思》中的两句诗:”
     - 固定后缀：“低头思故乡。”
     - 中间缺失 6 个 token，初始值为“床前明月光,”（局部最优）
     - 目标恢复“举头望明月,”（全局最优）

3. **APE Instruction Induction Benchmark**
   - 包含 **24 个子任务**，涵盖分类、推理、翻译等（如 Sentiment、Formality、Translation en-es 等）。
   - 每个任务随机选取 **6 个训练样本** 构造 energy 函数。
   - 使用原始 APE 测试集评估泛化性能。

### 实验设置与评估指标
| 设置项 | 描述 |
|-------|------|
| 模型 | Qwen2.5-0.5B-Instruct（诊断）、Qwen2.5-7B-Instruct（主实验） |
| Prompt 长度 | 固定长度（20 或 50 tokens） |
| 初始化 | 随机初始化（诊断任务）；APE prompts 初始化（主实验） |
| 评估指标 | **Test Accuracy (%)**（主要）、Training Energy（辅助分析） |
| 采样器参数 | GwL + PT，step size α=4.0，迭代次数 S=1000~4000 |
| 温度链数 | M+1 = 8~9 条链，温度范围 [1.0, 2.0] |

### 基线方法对比
- **Single-chain GwL**：无 PT 的消融版本
- **APE**：原始 prompt 作为 baseline
- **Random Sampling / Ancestral Sampling**：用于能量基准比较

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### 在 APE 指令诱导基准上的平均准确率提升
| 方法 | Average Test Accuracy |
|------|------------------------|
| APE（初始化） | 60.04% |
| **BayesPO (GwL+PT)** | **63.23%** |
| **绝对提升** | **+3.19 pp** |

> 表明 BayesPO 能有效改进已有 prompt，尤其当初始 prompt 存在语义漂移时。

#### 典型任务上的显著增益（部分）
| Task | APE Acc. | BayesPO Acc. | Δ |
|------|----------|-------------|----|
| Second Letter | 10.00% | 63.00% | +53.00 |
| Word in Context | 8.00% | 34.00% | +26.00 |
| Larger Animal | 51.00% | 66.00% | +15.00 |
| English-Spanish | 66.00% | 82.00% | +16.00 |
| Formality | 41.56% | 51.96% | +10.40 |

> 显示 BayesPO 特别擅长纠正语义偏差严重的初始 prompt。

### 与基线方法的对比结果
| 场景 | 结果 |
|------|------|
| **Single GwL vs PT**（诗歌补全） | 单链被困于“床前明月光,”；PT 成功转移至“举头望明月,” |
| **能量下降趋势** | 所有任务中训练能量持续下降，验证 sampler 有效性 |
| **失败案例** | Membership、Starting With 等任务出现过拟合，accuracy 下降 |

### 消融实验结果
| 组件 | 作用验证 |
|------|--------|
| **GwL + MH correction** | 在诊断任务中成功从随机 prompt 收敛到语义相关 prompt（如“古代汉字表达式”、“否定陈述”） |
| **Parallel Tempering** | 温度轨迹图显示频繁跨链交换，高温度链探索更广区域，证实其对跳出局部最优的关键作用 |
| **Non-weight-tied embedding 处理** | 方法可在 Qwen2.5 等现代模型上稳定运行，而标准 GwL 可能失效 |

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Prompt 优化可被形式化为贝叶斯后验采样问题**：
   - 通过定义 energy-based model，实现了 principled 的 prompt search 框架。
2. ✅ **Gradient-guided discrete MCMC + MH correction 可行**：
   - GwL 提案能有效利用 embedding 梯度引导 token 替换，MH 保障理论正确性。
3. ✅ **Parallel Tempering 显著改善全局探索**：
   - 在诗歌补全任务中，仅靠 PT 才能逃离局部最优，证明其必要性。
4. ✅ **Post-optimization 有效提升 APE prompts 性能**：
   - 平均提升 3.19%，最大单任务提升达 53%，尤其适用于初始 prompt 语义错位的任务。

### 方法的局限性
1. ⚠️ **小优化集下的过拟合风险（Energy-Accuracy Mismatch）**：
   - 如 Membership 任务：训练能量大幅下降（324→147），但测试 accuracy 从 81% 降至 49%。
   - 原因：优化过程拟合了训练样本中的虚假模式（如特定长度约束）。
2. ⚠️ **计算成本高昂**：
   - 当前实现约为 APE 的 **10 倍耗时**，因需多次 forward/backward + MH evaluation。
   - 不适合实时 prompt 调整，仅适用于高可靠性要求场景。
3. ⚠️ **固定 prompt 长度限制**：
   - 当前 sampler 不支持插入/删除操作，无法处理变长 prompt 编辑。

### 未来工作方向
1. 🔮 **加速 MCMC 收敛速度**：
   - 设计更高效的 proposal kernel 或引入 variance reduction 技术。
2. 🔮 **Prefix-Prediction 对齐的 LLM 训练目标**：
   - 探索在预训练中加入 instruction inversion 目标，使模型提供更有信息量的梯度。
3. 🔮 **Trans-dimensional MCMC**：
   - 引入 birth-death 或 reversible-jump moves，支持动态长度 prompt 优化。
4. 🔮 **Discrete Mini-batch MCMC with Valid Correction**：
   - 扩展至更大优化集，研究带 MH 校正的小批量采样策略，缓解过拟合。

---

> **总结**：  
> BayesPO 首次将 prompt optimization 建立在 **贝叶斯采样理论框架** 上，提出了一种 **principled、可解释、可收敛** 的优化路径。尽管当前存在计算开销大和小样本过拟合等问题，但它为 **概率化 prompt engineering** 开辟了新方向，是迈向 **可信 prompt 自动化** 的重要一步。

</details>

---

### 11. [qZACH-ViT: Quantization-Aware Intrinsic Explanations with Recursive Attribution-Stabilized Optimization](https://arxiv.org/abs/2607.15421)

**Authors**: Athanasios Angelakis  
**Category**: cs.LG  
**Published**: 2026-07-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.15421v1  

#### Abstract
Compact medical-image classifiers need efficiency and interpretable evidence, yet these goals are often addressed separately. We introduce qZACH-ViT, a quantization-aware extension of the zero-token (CLS-token-free), position-free ZACH-ViT backbone with recursive intrinsic patch-level class evidence...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：qZACH-ViT: Quantization-Aware Intrinsic Explanations with Recursive Attribution-Stabilized Optimization

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在医学图像分类任务中，模型需要同时满足**高效率**（compactness）和**可解释性**（interpretability）。然而，这两个目标通常被分开处理：
- 高效模型（如量化模型）可能在部署后改变内部证据路径，导致解释不稳定；
- 可解释性方法多为**post-hoc**（训练后解释），不参与预测过程，难以保证忠实性（faithfulness）；
- 量化（quantization）可能导致原始模型的决策依据发生偏移，影响临床可信度。

本文旨在解决：如何构建一个**既紧凑又内在可解释**，且在**INT8量化部署后仍能保持预测与解释一致性**的视觉模型。

---

### 提出的新方法与创新思路

#### （1）qZACH-ViT 架构
- **继承 ZACH-ViT**：零 CLS-token、无位置编码、全局平均池化聚合 patch 表示，参数量约 25万，适合低数据场景。
- **引入递归 patch 级类证据头（recursive patch-level class evidence）**：
  - 在两个 transformer block 后分别添加 evidence head，输出原始 logit 级 patch 证据 $ e^{(1)}, e^{(2)} $。
  - 最终 logit 是两阶段证据的加权平均：$ z = p \cdot \text{avg}(e^{(1)}) + (1-p) \cdot \text{avg}(e^{(2)}) $，其中 $ p=0.25 $。
  - **关键特性**：原始证据**精确重构 logit**（exact logit completeness），误差为0（浮点精度下验证）。
- **支持 W8A8 量化感知训练（QAT）**：
  - 所有线性投影层启用对称符号 INT8 伪量化（fake quantization）；
  - 训练时模拟量化噪声，使模型适应低精度环境。

#### （2）Recursive Attribution-Stabilized Optimization (RASO)
一种新的优化策略，用于联合优化分类与解释一致性目标：
- 分别计算分类损失梯度 $ g_c $ 和归因损失梯度 $ g_a $；
- 对 $ g_a $ 进行**范数匹配（norm-matching）**：$ u = \|g_c\|_2 \cdot g_a / (\|g_a\|_2 + \epsilon) $；
- 若 $ (g_c, u) < 0 $（即冲突），则**移除冲突分量**：
  $$
  u \leftarrow u - \frac{(u, g_c)}{\|g_c\|^2} g_c
  $$
- 将修正后的 $ u $ 加入 Adam 更新：$ g_{\text{RASO}} = g_c + \lambda_a u $
- **设计思想**：保护分类方向为主，仅允许解释梯度在正交或同向方向贡献，避免反向干扰。

---

### 相比现有方法的优势

| 维度 | 优势 |
|------|------|
| **架构设计** | 内在可解释性（intrinsic explanation），无需额外解释器；patch 证据直接构成 logit，具备数学完备性 |
| **部署可靠性** | 实际转换为 **executable mixed-precision ONNX INT8 图**，所有 16 个 MatMul 投影使用 `MatMulInteger` 操作符，经 ONNX Runtime 审计执行整数运算 |
| **解释稳定性** | RASO 显著提升解释在输入扰动下的鲁棒性和 sufficiency 性能 |
| **端到端性能** | INT8 模型不仅未降性能，反而在多数任务上**超越 FP32 基线**，实现“压缩增益” |

---

## 2. 核心实验方法和设置

### 数据集
使用 **MedMNIST v2** 中的 7 个 2D 医学图像子集：
- BloodMNIST（血细胞显微）
- PathMNIST（结直肠组织病理）
- BreastMNIST（乳腺超声）
- PneumoniaMNIST（胸部X光）
- DermaMNIST（皮肤镜）
- OCTMNIST（视网膜OCT）
- OrganAMNIST（腹部CT）

> 所有图像统一 resize 到 224×224，转为 RGB 输入。

---

### 实验设置

#### 训练协议
- **极低数据设定**：每类仅采样 **50 张训练图像**（from official train split）
- 固定 10 个随机种子：`[3,5,7,11,13,17,19,23,29,31]`
- Batch size: 16，Epochs: 23，Optimizer: Adam（lr=1e-4, no weight decay）
- 每个 dataset-seed 组合独立训练，共完成 **280 次训练运行**

#### 控制变量条件（4种）
| 条件 | 描述 |
|------|------|
| ZACH-ViT + Adam | FP32 基线，无内在解释机制 |
| qZACH-ViT + Adam | W8A8 QAT + 内在证据头，仅分类损失 |
| qZACH-ViT + Adam + Cattr | 上述 + 添加归因损失（scalar sum） |
| qZACH-ViT + RASO | 完整方法：QAT + 证据头 + RASO 优化 |

> 所有 qZACH-ViT 变体从相同初始状态开始，确保公平比较。

---

### 评估指标

#### 预测性能
- 多类任务：**Macro-F1**
- 二类任务：**AUC@0.5**（阈值0.5的ROC-AUC，沿用前序工作）
- 补充报告：probability AUROC、accuracy、Macro-F1

#### 部署保真度（Deployment Fidelity）
- **预测一致性**：source (FP32) vs deployed (INT8) 的标签一致率
- **主指标变化**：绝对差值（mean \|Δ\|）
- **最大 logit 扰动分析**：检查标签变化是否发生在决策边界附近

#### 解释保留性（Explanation Retention）
在 3,600 个 XAI 示例上评估：
- **Cosine similarity** of recursive attribution maps
- **Spearman rank correlation**
- **Top-10% overlap**
- **JS divergence**

#### 推理效率
- **序列化大小**：PyTorch checkpoint vs ONNX INT8 artifact
- **CPU 推理延迟**（batch size=1）：
  - 单线程 & 四线程下的 median latency
  - Speedup = PyTorch / ONNX

#### XAI 质量评估（source model）
- Deletion AUC、Insertion AUC
- Comprehensiveness、Sufficiency error
- SaCo（Transformer 特定忠实性指标）
- Occlusion rank correlation
- 输入噪声稳定性（cosine, JS）

---

### 基线方法对比
- **预测基线**：ZACH-ViT + Adam (FP32)
- **消融基线**：
  - qZACH-ViT + Adam（仅QAT）
  - qZACH-ViT + Adam + Cattr（普通损失加权）
- **XAI 方法对比**（same-model）：
  - Integrated Gradients
  - Token Grad-CAM
  - Attention Rollout-ZT
  - Gradient Attention Rollout-ZT
  - RISE
  - Patch Occlusion
  - Random Attribution（负控）

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 指标 | 数值 |
|------|------|
| **总训练次数** | 280 runs |
| **INT8 转换成功数** | 210 / 210（全部成功） |
| **预测一致性（source → INT8）** | **99.9751%** |
| **主指标平均绝对变化** | 0.000133 |
| **最大主指标变化** | 0.004386 |
| **解释图平均余弦相似度** | **0.999955** |
| **平均 Spearman 秩相关** | 0.9944 |
| **Top-10% overlap 平均值** | 0.9692 |
| **ONNX 序列化体积缩减** | **70.0%**（0.988 MiB → 0.296 MiB） |
| **单线程推理加速比** | **1.41×** |
| **四线程推理加速比** | **2.39×** |

---

### 与基线方法对比结果

#### （1）预测性能（Table 1 & Figure 1）
- 所有 **qZACH-ViT 变体在 7 个数据集上的平均性能均超过 FP32 基线**；
- **RASO 效果最强**：
  - 平均增益：**+0.0368**（vs baseline）
  - Adam：+0.0313，Loss-only control：+0.0309
- 成对比较中，RASO 在 **57/70 种 seed-dataset 组合中优于 baseline**（Adam 为 53/70）

#### （2）部署保真度（Table 2 & A.3）
- **仅 240 个样本在 INT8 转换后标签改变**（out of 964,920）
- 所有标签变化均发生在源模型决策边界附近（margin ≤ 2× max logit error）
- **RASO 改变最少**：仅 **56 个错误**（Adam: 95, Loss-only: 89）

#### （3）解释保留性（Table 2）
- 所有条件解释高度保留：
  - 平均 cosine ≥ 0.999948
  - RASO 最佳：**0.999968**
- Top-10% overlap ≥ 0.9684
- RASO 的 JS divergence 最小（1.13e-5）

---

### 消融实验结果

#### RASO vs 普通损失加权（qZACH-ViT + Adam + Cattr）

| 指标 | RASO vs Loss-only 差异 | 是否显著（Holm校正） |
|------|------------------------|---------------------|
| Sufficiency error ↓ | **-0.0300** | ✅ Yes ($p=0.0111$) |
| Noise cosine ↑ | **+0.00177** | ✅ Yes ($p<0.0001$) |
| Noise JS ↓ | **-0.00055** | ✅ Yes ($p<0.0001$) |
| 其他7项（如 Insertion, SaCo） | 无显著差异 | ❌ No |

> **结论**：RASO 显著提升了**解释稳定性与充分性**，但并非在所有 XAI 指标上占优。

#### 梯度冲突频率
- 平均有 **45.9% 的训练 batch 出现分类与归因梯度冲突**
- 最高达 53.4%（OrganAMNIST），最低 37.7%（OCTMNIST）
> 表明 RASO 的梯度修正机制是活跃且必要的。

---

## 4. 关键结论和发现

### 主要发现

✅ **qZACH-ViT 是真正可部署的紧凑内在可解释模型**：
- 成功将 QAT 与内在解释结合；
- 所有 210 个 checkpoint 转换为含 **16 个 signed INT8 MatMulInteger 节点**的 ONNX 图；
- ONNX Runtime 实际执行整数矩阵乘法，通过部署审计。

✅ **INT8 模型性能反超 FP32 基线**：
- 不是“无损压缩”，而是“压缩增益”；
- 可能源于 QAT 正则化效应或 RASO 稳定训练动态。

✅ **解释路径高度稳定保留**：
- 解释图在 INT8 转换后几乎不变（cosine > 0.9999）；
- 支持“一次训练，多端部署”的可信推理流程。

✅ **RASO 是有针对性的优化改进**：
- 显著提升 sufficiency 与噪声稳定性；
- 不追求全面领先，而是聚焦保护分类方向的同时增强解释一致性。

✅ **内在解释 ≠ 最强 post-hoc 方法**：
- Gradient Attention Rollout-ZT 在 deletion、insertion、SaCo 上更强；
- 但其 XAI 延迟高达 **22.4 ms/image**，而 qZACH intrinsic 仅 **9.9 ms**；
- **优势在于无需 backward pass，天然支持边缘部署**。

---

### 局限性（Limitations）

1. **混合精度而非全整数推理**  
   LayerNorm、softmax、residual add、dequantization 等仍为 FP32。

2. **运行时栈混杂**  
   延迟比较基于 PyTorch eager vs ONNX Runtime，包含框架差异，非纯 kernel 对比。

3. **缺乏硬件元数据**  
   未记录具体 CPU 型号，限制结果泛化性。

4. **无峰值内存声明**  
   仅提供序列化大小，未测量运行时内存占用。

5. **无定位标注（no ground-truth localization）**  
   MedMNIST 仅有分类标签，无法进行 Dice 或 IoU 等解剖级评估。

6. **单一基准族**  
   所有任务来自 MedMNIST，需外部数据验证迁移能力。

7. **粗粒度 patch（14×14）**  
   限制空间分辨率，上采样热力图可能误导临床解读。

8. **RASO 缺少与其他多目标优化器的直接对比**  
   如 PCGrad、CAGrad、GradNorm 等，留待后续工作。

9. **框架移植偏差**  
   基线为 PyTorch 实现，非完全复现原 TensorFlow 结果（但内部比较受控）。

10. **无临床主张**  
    热力图反映模型关注区域，**不代表病理因果关系**，需专家验证。

---

### 未来工作方向

- 扩展至 **3D 医学影像**（如 MedMNIST-3D）
- 实现 **fully integer-only inference**，包括 LayerNorm 和 softmax 的整数量化
- 引入 **expert-annotated localization 数据集** 进行临床可信评估
- 开展 **multi-site external validation**
- 探索 **RASO 与其他 MTL 优化器的系统比较**
- 发布 **mobile/edge 端部署 demo**（如 Android NNAPI、Core ML）
- 结合 **concept bottleneck modeling** 提升高层语义可解释性

---

> **最终结论**：  
> qZACH-ViT 与 RASO 共同构成了一个**经过严格部署验证的、紧凑、高效、内在可解释且解释稳定的医学图像分类解决方案**。它不仅在技术上实现了“预测+解释+量化”的三重统一，更通过大规模实证研究展示了**真实部署中的高保真性与实用性**，为可信 AI 在医疗领域的落地提供了坚实基础。

</details>

---

### 12. [Knowledge-Guided Cross-Modal Fusion for Adult-to-Pediatric ECG Transfer via Label-Conditioned Contrastive Alignment](https://arxiv.org/abs/2607.15928)

**Authors**: Xinran Liu, Yuwen Li, Hongxiang Gao, Heyang Xu, Jianqing Li, Zongmin Wang, Chengyu Liu  
**Category**: cs.LG  
**Published**: 2026-07-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.15928v1  

#### Abstract
Adult and pediatric electrocardiogram (ECG) interpretation relies on age-sensitive criteria, and models pretrained mainly on adult ECGs often transfer poorly to pediatric populations when pediatric labels are scarce. Existing multimodal ECG--text methods typically align waveforms and text at the glo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Knowledge-Guided Cross-Modal Fusion for Adult-to-Pediatric ECG Transfer via Label-Conditioned Contrastive Alignment

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该研究针对**成人到儿科 ECG 模型迁移中的“transfer gap”问题**。由于儿童与成人在心率、间期、电压、传导模式等生理标准上存在显著差异，直接将在大规模成人 ECG 数据（如 MIMIC-IV）上预训练的模型迁移到儿科人群时表现不佳，尤其是在儿科标注数据稀缺的情况下。

此外，现有跨模态 ECG-文本对齐方法通常采用全局样本级融合（global sample-level alignment），导致多个并发诊断的证据混淆，且多数方法在推理阶段依赖文本输入，不适用于临床实际部署。

---

### 提出的新方法：PEACE 框架
作者提出 **PEACE**（Pediatric-Adult ECG Alignment via Cross-modal Enhancement），一个**知识引导的跨模态融合框架**，其核心思想是：

- 利用**结构化临床知识作为特权监督信号**（privileged information），仅用于训练阶段；
- 推理时**仅使用 ECG 信号**，无需额外文本输入，符合临床实用性要求；
- 通过**标签条件化的对比对齐机制**实现更精细的知识融合。

#### 核心组件：
1. **Label Query Network (LQN)**  
   将每个诊断标签作为 query，分别对 ECG tokens 和三个轴向（rhythm, morphology, ST-T）的知识 tokens 进行 cross-attention，实现**按标签探针特征**，避免多标签纠缠。

2. **Label Set Aware Bidirectional Contrastive Learning (LSBC)**  
   当两个记录共享至少一个诊断标签时，将其 ECG 表征与由正标签组合而成的 fused knowledge embedding 对齐，提升语义一致性。

3. **Curriculum Adaptive Fusion (CAF)**  
   动态调节 LSBC 对齐强度：早期训练抑制对齐以稳定优化；随着分类损失下降和训练进度推进，逐步增强对齐作用。

4. **Axis-aligned Knowledge Composition**  
   每个诊断类别的知识被组织为三个独立轴（rhythm, morphology, ST-T），而非单一文本描述，保留了解剖/病理维度结构。

---

### 相比现有方法的优势
| 维度 | PEACE | 传统方法 |
|------|-------|--------|
| 融合粒度 | **标签级别**，区分不同病因路径 | 全局样本级融合，易混杂证据 |
| 知识结构 | 显式建模 rhythm/morphology/ST-T 三轴 | 单一文本嵌入或自由文本 |
| 推理依赖 | **仅需 ECG 信号** | 多数需测试时提供报告或文本 |
| 迁移效率 | 在极少量儿科标注下（如 50-shot）性能跃升 | 需大量标注才能有效微调 |
| 对比学习设计 | 基于共享标签的双向对比（LSBC） | 通常基于样本对或单向对齐 |

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 描述 |
|--------|------|
| **MIMIC-IV-ECG** | 预训练数据：超过 80 万条成人 ECG 记录，用于源域初始化 |
| **ZZU-pECG** | 主要目标域：包含 11,643 名儿童（0–14岁），共 7,593 条有效记录，用于评估儿科迁移能力 |
| **PTB-XL** | 成人主导数据集（含少量儿科），用于验证在非儿科任务上的泛化性 |

所有 ECG 统一为 10 秒、12 导联、500Hz，并进行带通滤波和归一化处理。

---

### 实验设置与评估协议

#### 三种迁移场景：
| 场景 | 设置说明 |
|------|---------|
| **Zero-shot Transfer** | 冻结预训练模型，直接在 ZZU-pECG 上测试，无任何儿科微调 |
| **50-shot Adaptation** | 每类最多取 50 个阳性样本进行微调（低资源设定） |
| **Full Fine-tuning** | 使用完整儿科训练集进行微调 |

#### 评估指标
- **主指标**：macro average AUC（阈值无关，反映排序能力）
- **辅助指标**：macro BAcc（balanced accuracy）、macro F1（均经验证集调参后固定阈值）
- 所有结果在 patient-level 分割下报告（train/val/test = 8:1:1）

#### 基线方法分组对比
| 类别 | 包括的方法 |
|------|-----------|
| **Group A**: 领域自适应与简单融合 | DANN, MMD, Early fusion, Late fusion |
| **Group B**: ECG 基础模型与知识预训练 | ST-MEM, MERL, KED |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| 方法 | Zero-shot AUC (%) | 50-shot AUC (%) | Full FT AUC (%) | PTB-XL AUC (%) |
|------|------------------|----------------|------------------|----------------|
| **PEACE (Ours)** | **59.39** | **81.74** | **91.56** | **96.90** |
| DANN (best in Group A) | 49.33 | 81.70 | 89.67 | 96.54 |
| MERL (best in Group B) | 58.58 | 70.65 | 80.97 | 92.49 |

> 注：PTB-XL 结果基于九个共有标签的 macro AUC，不可与 ZZU-pECG 完全对标。

---

### 与基线方法的对比结果

#### ✅ 对比 Group A（领域自适应/简单融合）
- **Zero-shot**：PEACE 比 DANN/MMD 提高 **+10.06 pp**，表明其更强的可迁移初始化；
- **50-shot**：与 DANN 相当（81.74 vs 81.70），但优于其他融合方法（Early/Late fusion ≤80.7）；
- **Full FT**：领先 **+1.62 pp**，显示持续增益；
- **结论**：即使面对强领域适配初始化，PEACE 仍能取得更好迁移效果。

#### ✅ 对比 Group B（基础模型/知识预训练）
- **Zero-shot**：小幅领先 MERL/KED（+0.8~1.6 pp）；
- **50-shot**：大幅领先 **+11.09 pp**（vs MERL），体现卓越的小样本适应能力；
- **Full FT**：领先 **+10.01 pp**，说明知识对齐策略显著增强了表征质量；
- 特别地，KED 在 50-shot 下性能反而下降（57.77 → 50.66），而 PEACE 稳定上升。

---

### 消融实验结果（Table 3）

| 配置 | Zero-shot AUC | 50-shot AUC | Full FT AUC |
|------|--------------|-------------|------------|
| **Full PEACE** | **59.39** | **81.74** | **91.56** |
| No text | 46.01 | 66.37 | 66.86 |
| Without LSBC | 56.23 | 77.74 | 89.63 |
| Without CAF | 47.10 | 77.69 | 90.45 |
| Single text (fused descriptor) | 49.20 | 74.30 | 88.70 |

#### 消融分析结论：
- **移除文本分支（No text）** 导致全面崩溃，证明知识监督的关键作用；
- **移除 LSBC** 显著降低小样本性能（↓4.0 AUC @50-shot），说明标签感知对比对齐至关重要；
- **移除 CAF** 导致 zero-shot 性能骤降（↓12.3），表明课程门控对稳定训练和高质量初始化极为重要；
- **使用单一融合描述符** 而非三轴结构也造成性能下降，验证了 axis-separated knowledge 设计的有效性。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **标签条件化的知识对齐（而非全局融合）是提升儿科迁移性能的核心驱动力**；
2. ✅ PEACE 在**有限标注条件下（few-shot）优势最明显**，适合现实世界中儿科数据稀疏的应用场景；
3. ✅ 三轴结构化知识表示（rhythm/morphology/ST-T）有助于解耦病理机制，提升模型可解释性和迁移鲁棒性；
4. ✅ CAF 机制有效平衡了早期训练稳定性与后期知识注入强度，提升了整体迁移效率；
5. ✅ 推理阶段完全脱离文本输入，具备良好的临床部署潜力。

---

### 方法的局限性
1. ❌ 使用的临床知识描述由 **Gemini 自动生成并人工修订**，虽经专家审核，但仍为通用规则，**未包含年龄特异性儿科标准**；
2. ❌ 所有知识来源于训练标签，因此 **knowledge branch 不可用于测试时诊断输出**；
3. ❌ Grad-CAM++ 可视化仅为启发式展示，**未建立严格的归因因果性**；
4. ❌ 缺乏显式的发育阶段建模（如新生儿 vs 学龄前），未来可引入连续年龄编码；
5. ❌ DANN/MMD 的 zero-shot 对比未使用无监督领域适配步骤，仅为冻结初始化比较。

---

### 未来工作方向
1. 🔄 开发**真正的年龄感知知识描述系统**，整合儿科各年龄段的动态 ECG 标准；
2. 🔍 构建**专家评审的权威儿科 ECG 知识库**，替代当前生成式描述；
3. 🧠 引入**连续 age-aware transfer learning 机制**，支持从胎儿到青少年的平滑过渡；
4. 💬 探索将 axis-structured knowledge 用于构建**面向医生的知识辅助界面**，支持“专家-in-the-loop”决策；
5. 📈 将本框架扩展至其他医学信号（如 EEG、PPG）的跨人群迁移任务。

--- 

> **总结一句话**：  
> PEACE 通过**标签引导的结构化知识对齐 + 渐进式融合控制**，实现了高效、实用、可解释的成人到儿科 ECG 模型迁移，在极低标注成本下展现出显著优势，为知识驱动的医疗 AI 提供了新范式。

</details>

---

### 13. [Adaptive Multi-Step Lookahead Decoding for Diffusion Language Models](https://arxiv.org/abs/2607.15655)

**Authors**: Yingqian Cui, Wei Deng, Lantao Mei, Hang Li, Charu C. Aggarwal, Hui Liu, Yue Xing  
**Category**: cs.CL  
**Published**: 2026-07-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.15655v1  

#### Abstract
Masked diffusion language models (DLMs) enable parallel text generation by iteratively refining masked tokens, offering a promising alternative to autoregressive decoding. Recent lookahead-based decoding methods improve the accuracy--efficiency trade-off by exploring future decoding states before co...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Adaptive Multi-Step Lookahead Decoding for Diffusion Language Models**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
现有的 **Masked Diffusion Language Models (DLMs)** 虽然支持并行文本生成，但其推理策略中的 **lookahead decoding** 方法大多局限于 **one-step lookahead**，即仅基于当前解码状态下一步的信息增益来选择 token 提交。这种局部探索方式在长程规划中可能陷入次优路径。

此外，简单地扩展为固定深度的多步 rollout（如 naive 2-step）会带来额外计算开销，且无法适应不同样本或解码阶段的异质性需求，导致效率与质量权衡不佳。

### 🚀 提出的新方法：AdaLook
本文提出 **AdaLook**（Adaptive Lookahead Decoding），一种**自适应多步前瞻解码框架**，具有以下两个核心机制：

1. **Adaptive Rollout Continuation（自适应 rollout 继续）**  
   在每一步 lookahead 后，通过计算候选路径得分的**方差**（variance）判断是否继续 rollout：
   - 若方差低于阈值 $T$，说明候选路径尚不具区分度，需进一步展开；
   - 否则提前终止，避免不必要的深层 rollout。

2. **Dynamic Branch Expansion（动态分支扩展）**  
   多步 rollout 过程中，对每个假设路径独立判断是否需要重新触发 lookahead 探索：
   - **Case 1（无分支需扩展）**：继续 rollout；
   - **Case 2（所有分支都需扩展）**：选择最优路径状态返回主循环，作为下一轮 lookahead 的起点；
   - **Case 3（混合状态）**：剪枝仍需扩展的分支，保留稳定分支继续搜索。

该设计实现了：
- 避免固定深度带来的冗余计算；
- 支持从中间状态“重触发”lookahead，增强探索灵活性；
- 更好地捕捉长期收益高的解码轨迹。

### 🔍 相比现有方法的优势
| 方法 | 局限性 | AdaLook 的改进 |
|------|--------|----------------|
| One-step Lookahead (e.g., ETE) | 仅优化即时信息增益，易陷局部最优 | 引入多步探索，考虑更长远影响 |
| Naive Multi-step Lookahead | 固定深度，计算成本高，适应性差 | 自适应决定 rollout 深度，节省资源 |
| Confidence-aware Decoding (e.g., Fast-dLLM) | 缺乏主动探索机制 | 显式建模未来状态价值 |

> ✅ **核心优势**：在相近甚至更少 decoding steps 下实现更高 accuracy，尤其在复杂任务上表现显著提升。

---

## 2. **核心实验方法和设置**

### 📚 使用的数据集
实验覆盖多个主流 benchmark，涵盖通用知识与推理能力：
- **MMLU**（500 samples）：多学科知识理解
- **GSM8K**（1,319 samples）：小学数学应用题
- **MATH500**（500 samples）：高中及以上难度数学题
- **BBH**（500 samples）：挑战性思维链任务
- **HumanEval**（代码生成任务，用于分析特殊场景）

### ⚙️ 实验设置
- **模型**：
  - 主要使用 **LLaDA-8B-Instruct**（Nie et al., 2026）
  - 补充实验使用 **Dream-v0-Instruct-7B**（Ye et al., 2025）
- **生成参数**：
  - 序列长度：512
  - Block size：64
  - 解码方式：greedy decoding
  - 硬件：NGPII NVIDIA H200 / B200 GPUs
- **评估指标**：
  - **Accuracy**：任务正确率
  - **Decoding Steps**：平均前向传播次数（衡量效率）
  - **Latency**：端到端生成时间（ms/step）

### 🆚 基线方法对比
| 方法 | 类型 | 来源 |
|------|------|------|
| **Fast-dLLM** | Confidence-aware 并行解码 | Wu et al., 2025 |
| **ETE (Explore-then-Exploit)** | One-step lookahead | Fu et al., 2025 |
| **AdaLook (ours)** | Adaptive multi-step lookahead | 本文提出 |

> 所有方法共享相同的 hyperparameter calibration 规则（见下文），确保公平比较。

---

## 3. **主要实验结果和性能指标**

### 📈 关键性能数据（以 MATH500 为例）
| 方法 | 最大 Accuracy | 达到 ~45% Acc 所需 Steps |
|------|---------------|--------------------------|
| Fast-dLLM | 42.2% | ~70 steps |
| ETE (Optimized) | 42.6% | ~55 steps |
| **AdaLook (Optimized)** | **43.6%** | **~45 steps** |

> ➤ 在约 **45 步**时，AdaLook 比 ETE 高出 **约 4.5% 绝对准确率**。

### 📊 整体趋势（Figure 3）
- 在 **MATH、BBH、MMLU** 等高难度任务上，AdaLook 显著优于所有 baseline；
- 在较简单的 **GSM8K** 上也有提升，但幅度较小；
- **AdaLook (Standard)** 即使未调优 $\gamma$，也普遍优于 ETE (Optimized)，显示强鲁棒性。

### 🔬 消融与分析实验

#### （1）Latency 分析（Table 1）
| Beam Size | H200 上 AdaLook vs ETE 延迟增加 |
|-----------|-------------------------------|
| k=2 | +5.7% |
| k=4 | +18.6% |
| k=6 | +19.2% |

但在更强硬件（如 B200）上，k=4 时仅增加 **~7% 延迟/step**，表明实际开销可控。

> 💡 结论：**适度 beam size 下，额外延迟可忽略不计**。

#### （2）Code Generation 场景分析（HumanEval）
- 在 **HumanEval** 上，无论是 ETE 还是 AdaLook，相比 Fast Block Sampling 提升极小；
- 推测原因：代码生成依赖**长距离结构依赖**，短 horizon 的 lookahead 得分信号弱，难以有效指导探索。

> ➤ 表明 AdaLook 对**局部可分解性强的任务更有效**（如数学推理），而对全局结构敏感任务仍有局限。

#### （3）Hyperparameter Calibration 发现
- $C^*(N)$（confidence threshold）随 $N$（block budget）增大而缓慢上升，经验规律为：
  $$
  C(N) \in [0.5 + 0.05\log_2 N,\ 0.5 + 0.10\log_2 N]
  $$
- 理论解释：当 $N$ 大时，block 内有更多 refine 步骤，可用更高置信度提交 token。

---

## 4. **关键结论和发现**

### ✅ 主要结论
1. **One-step lookahead 存在局限性**：虽能提升效率，但因视野短，易错过全局更优路径。
2. **Naive multi-step rollout 不够高效**：固定深度引入冗余计算，且不能灵活响应中间状态不确定性。
3. **AdaLook 实现更优 trade-off**：
   - 动态控制 rollout 深度 → 减少无效计算；
   - 支持中间状态重触发 lookahead → 提升探索质量；
   - 在多个 benchmark 和 backbone 上一致超越 baseline。
4. **越难的任务，收益越大**：在 MATH、BBH 等高不确定性任务中，AdaLook 提升最明显。

### ⚠️ 方法的局限性
1. **对 code generation 类任务效果有限**：受限于 rollout horizon，难以捕获长程语义依赖。
2. **每步 latency 略高**：由于 batched forward passes，beam size 增大会轻微增加延迟（但现代 GPU 可缓解）。
3. **依赖 confidence score 质量**：若模型 confidence 不准（calibration 差），会影响 rollout 判断。

### 🔮 未来工作方向
1. 设计更适合 **code generation** 的 lookahead 机制（例如结合 syntax-aware reward）；
2. 将 AdaLook 与 **KV Caching** 技术结合，进一步降低 batched forward 成本；
3. 探索 **learnable rollout policy** 替代 hand-designed variance gating；
4. 扩展至 **multimodal diffusion models** 中的联合生成任务。

---

> ✅ **一句话总结**：  
> **AdaLook 通过自适应多步前瞻机制，在几乎不增加实际延迟的前提下，显著提升了 DLMs 的 accuracy-efficiency trade-off，尤其适用于高难度推理任务。**

</details>

---

### 14. [Physics-Based Deep Spatiotemporal Hyperlocal Radar Nowcasting with a Multi-Variable U-Net for High-Resolution Precipitation Forecasting](https://arxiv.org/abs/2607.16080)

**Authors**: Akshay Sunil, Muhammed Rashid, Raja Sekhar Sivaraju, Sushma Nair, Subimal Ghosh  
**Category**: cs.LG  
**Published**: 2026-07-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.16080v1  

#### Abstract
Precipitation nowcasting over the immediate 10-90 min period is important for flood management and real-time decision-making in urban regions. Conventional short-range forecasting with high-resolution numerical weather prediction requires frequent data assimilation, model initialization, and spin-up...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Physics-Based Deep Spatiotemporal Hyperlocal Radar Nowcasting with a Multi-Variable U-Net for High-Resolution Precipitation Forecasting*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该研究针对**城市地区短时强降水（nowcasting）预测困难**的问题，尤其是在像印度孟买这样的沿海季风气候区。传统基于**Numerical Weather Prediction (NWP)** 的方法因计算延迟（data assimilation、model spin-up）难以满足实时决策需求；而传统的雷达外推法（如TREC、optical flow）在超过30–60分钟的lead time后迅速退化，无法捕捉对流的**生成、增强与衰减**过程。

此外，大多数深度学习模型仅依赖**reflectivity**，忽略了**Doppler radial velocity**中蕴含的关键动力学信息（如辐合、切变），且缺乏可解释性，限制了其在业务预报中的可信度。

---

### 🚀 提出的新方法与创新点

1. **多变量输入设计（Multi-Variable Input Engineering）**  
   首次将**multi-elevation reflectivity**、**Doppler radial velocity**及其**gradient-derived proxy features**联合输入U-Net架构，构建了物理感知的输入张量。这些proxy特征包括：
   - **Velocity magnitude** $|V_r|$
   - **Divergence-like proxy**: $S = \frac{\partial V_r}{\partial x} + \frac{\partial V_r}{\partial y}$
   - **Directional shear proxy**: $\theta = \text{atan2}(\frac{\partial V_r}{\partial y}, \frac{\partial V_r}{\partial x}) / \pi$
   - **Vorticity-like proxy**: $\omega = \frac{\partial V_r}{\partial y} - \frac{\partial V_r}{\partial x}$

   这些特征无需进行完整的风场反演（full wind retrieval），即可编码低层辐合、边界相互作用等有利于对流触发的动力信号。

2. **高反射率注意力机制（High-Reflectivity Attention, HRA）**  
   在encoder bottleneck和decoder末端引入注意力模块，增强网络对**convective cores**（高dBZ区域）的敏感性，提升强降水预测能力。

3. **物理引导的可解释性分析（Physics-Guided Attribution）**  
   使用feature attribution技术验证模型是否关注气象上有意义的结构（如辐合线、阵风锋），提高模型在业务应用中的可信度。

4. **紧凑高效的确定性框架**  
   模型为**feed-forward deterministic U-Net**，训练后可在普通计算机上**秒级生成预测**，适合每7.5分钟一次的实时更新周期。

---

### 🔍 相比现有方法的优势

| 方法类型 | 局限性 | 本文优势 |
|--------|-------|---------|
| Optical Flow / Persistence | 无法建模对流演化，>60 min技能骤降 | 可学习非线性发展，长lead time表现更优 |
| NWP / Convection-Allowing Models | 计算昂贵，初始化延迟大 | 实时性强，无spin-up延迟 |
| DGMR / Generative Models | 复杂、资源消耗大 | 轻量化、部署成本低 |
| Reflectivity-only DL models | 忽略动力信息 | 引入velocity gradient proxy，提升物理一致性 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- **来源**：印度气象局（IMD）位于Veravali（~19.1°N, 72.9°E）的C波段Doppler天气雷达
- **时间范围**：2023年5月至8月（主季风期）
- **空间范围**：以孟买为中心，250 km半径内，重采样至**1 km × 1 km Cartesian网格（501×501 → 128×128）**
- **变量**：
  - 多仰角（10层）**reflectivity (Z)** 和 **radial velocity (Vr)**
  - 构造6通道/仰角的输入特征（共 $6 \times 9 = 54$ 输入通道）
- **目标输出**：未来12帧（每7.5分钟一步）的**composite reflectivity（MAXZ）**，最长lead time达90分钟

---

### ⚙️ 实验设置
- **训练/验证/测试划分**：80%/10%/10%，按时间分割（temporally disjoint）
- **输入分辨率**：128×128，9个仰角层
- **输出序列长度**：12步（7.5–90 min）
- **网络架构**：Encoder-Decoder U-Net，含skip connections、bilinear upsampling
- **优化器**：Adam ($lr = 1\times10^{-4}$)，early stopping基于validation SSIM
- **硬件**：NVIDIA GPU（CUDA-enabled），batch size = 8

---

### 📏 评估指标

#### ✅ 分类指标（Categorical Metrics）
用于评估≥10, ≥20, ≥30 dBZ事件的预测能力：
- **Critical Success Index (CSI)**
- **Equitable Threat Score (ETS)**
- **Probability of Detection (POD)**
- **False Alarm Ratio (FAR)**
- **Frequency Bias**

#### ✅ 连续指标（Continuous Metrics）
- **RMSE**（归一化空间平均）
- **Spatial Correlation**
- **Structural Similarity Index Measure (SSIM)**

#### ✅ 结构分析
- **Power Spectral Density (PSD)**：评估不同尺度的空间结构保留情况

#### ✅ 可解释性
- **Feature Attribution Maps**：检查模型是否关注物理合理的前兆结构

---

### 🆚 基线方法对比
- **Persistence**（最简单基线）：假设当前回波保持不变
- **Optical Flow Extrapolation**：基于光流法的运动场外推

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（90分钟lead time）

| Threshold | CSI   | ETS   | POD   | FAR   | Bias  |
|----------|-------|-------|-------|-------|-------|
| ≥10 dBZ  | 0.437 | 0.314 | 0.70  | 0.41  | 1.23  |
| ≥20 dBZ  | 0.332 | 0.367 | 0.65  | 0.38  | 1.23  |
| ≥30 dBZ  | 0.193 | 0.175 | 0.45  | 0.30  | 0.49  |

> 注：Bias < 1 表示**系统性低估**强对流核。

---

### 🔁 与基线方法对比

#### ✅ 分类技能（CSI）
- 在所有阈值下，模型在**>30 min lead time后全面超越persistence**
- 例如在≥20 dBZ、90 min时：
  - 模型 CSI = **0.332**
  - Persistence CSI = **~0.28**
- 对于弱降水（≥10 dBZ），模型保持较高POD（>0.7）且FAR可控（<0.4）

#### ✅ 连续技能
| Lead Time | RMSE (Model) | RMSE (Persistence) | Correlation (Model) | Correlation (Persistence) |
|----------|---------------|------------------------|------------------------|------------------------------|
| 7.5 min  | 3.411         | **2.315**              | 0.846                 | **0.897**                    |
| ~31 min  | **4.275**     | 4.310                  | **0.774**             | 0.748                        |
| ~85 min  | **4.746**     | 5.461                  | **0.714**             | 0.629                        |

> 💡 **关键发现**：尽管persistence在短lead time占优，但本模型在**>30 min后实现更低RMSE和更高空间相关性**，表明其能更好捕捉风暴演变结构。

---

### 🔍 消融实验与分析（隐含在文中）

虽然未明确列出消融表，但从方法设计和讨论中可推断以下结论：

1. **多变量输入优于reflectivity-only baseline**  
   - 引入radial velocity及其gradient proxy显著提升了对**对流增长和中高强度回波**的表示能力
   - 特别是在捕捉**边界触发机制**方面更具优势

2. **HRA模块有效增强对强对流的敏感性**  
   - 注意力机制使网络聚焦于高dBZ区域，缓解了确定性模型普遍存在的“模糊化”倾向

3. **PSD谱分析显示结构退化模式合理**  
   - 预测随lead time增加逐渐丢失小尺度功率（<10 km），但大尺度结构保持较好
   - 错误主要表现为**intensity attenuation**而非虚假回波泛滥，具有操作安全性

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **融合多变量雷达观测可显著提升nowcasting技能**  
   利用Doppler radial velocity梯度构造的**proxy kinematic features**（如divergence-like, vorticity-like）能有效编码对流触发前兆信号，无需复杂风场反演。

2. **模型在长lead time（>60 min）优于persistence**  
   尽管短时（<30 min）不如外推法，但在**30–90 min窗口内表现出更强的时空一致性与结构保真度**。

3. **误差行为保守且具操作价值**  
   - 主要误差是**低估强对流核强度**（underprediction），而非大量虚警
   - False Positive Rate（FPR）始终较低，适合用于**城市防洪预警、航空调度**等对误报敏感的应用

4. **可解释性分析支持物理合理性**  
   Attribution maps显示模型关注**辐合线、边界层阵风锋、剪切区**等典型对流触发结构，增强了业务人员信任。

---

### ⚠️ 方法的局限性

1. **对极端强对流（>30 dBZ）预测能力有限**  
   - 90 min CSI仅为0.193，且Bias下降至0.49，说明**compact high-reflectivity cores被严重平滑**
   - 源于确定性point-wise loss倾向于“平均化”罕见事件

2. **确定性框架无法量化不确定性**  
   缺乏概率输出，在极端事件风险评估中存在不足

3. **依赖高质量雷达数据**  
   对dealiasing、clutter removal等预处理步骤敏感，在数据质量差区域性能可能下降

---

### 🔮 未来工作方向

1. **发展混合确定-随机框架（Hybrid Deterministic-Stochastic）**  
   如将当前模型作为generator嵌入GAN或diffusion架构，以生成更尖锐的极端回波。

2. **引入显式物理约束**  
   加入mass continuity、transport equation等物理正则项（类似NowcastNet、LUPIN），提升长期稳定性。

3. **改进损失函数以强调极端事件**  
   设计scale-aware或event-focused loss，结合focal loss思想，提升对稀有强对流的学习能力。

4. **扩展至多城市或多气候区验证**  
   验证该proxy-feature方法在其他季风或热带城市的普适性。

---

> **总结一句话**：  
> 本文提出了一种**轻量、高效、物理感知的雷达nowcasting U-Net框架**，通过融合**multi-variable Doppler特征**与**attention机制**，实现了在孟买季风环境下优于persistence的90分钟高分辨率降水预测，并通过可解释性分析增强了业务可信度，为城市实时防灾提供了实用工具。

</details>

---

### 15. [Improving Improved Kernel PLS](https://arxiv.org/abs/2607.16138)

**Authors**: Ole-Christian Galbo Engstr{\o}m  
**Category**: cs.LG  
**Published**: 2026-07-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.16138v1  

#### Abstract
Improved Kernel Partial Least Squares (IKPLS) algorithms 1 and 2 are among the fastest PLS calibration algorithms. This article focuses on two shared steps, the computation of the $\mathbf{X}$ rotations, $\mathbf{R}$, and the $\mathbf{Y}$ loadings, $\mathbf{Q}$, and accelerates both. For $\mathbf{R}...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Improving Improved Kernel PLS —— 核心结论与实验结果总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文针对 **Improved Kernel PLS (IKPLS)** 算法中的两个计算瓶颈步骤进行了优化：
- **X rotations $ R $** 的计算依赖于逐项累加（term-by-term accumulation），效率低且难以并行。
- **Y loadings $ Q $** 的计算在多数情况下复杂度较高，尤其是当响应变量数量 $ M $ 较小时，存在冗余计算。

尽管 IKPLS 已是目前最快的 PLS 算法之一，但在大规模数据或 GPU 加速场景下仍有进一步提速空间。

---

### 提出了什么新方法或新思路

#### （1）对 $ R $ 的改进：直接矩阵乘法替代累加
提出使用 **直接评估策略（direct evaluation strategy）** 替代原始的递推累加公式：

$$
\mathbf{r}_a = \mathbf{w}_a - R_{a-1}(P_{a-1}^\top \mathbf{w}_a)
$$

该方法避免了串行累加，转而利用已累积的 $ R_{a-1} $ 和 $ P_{a-1} $ 进行一次矩阵-向量乘法，显著提升现代硬件上的并行执行效率。

#### （2）对 $ Q $ 的改进：重用已有中间量
首次从数学上证明，在以下两种情形中，$ \mathbf{q}_a $ 可由第 2 步中已计算的中间量直接导出（仅差一个常数因子）：
- $ M = 1 $（即 PLS1）
- $ 2 \leq M < K $（即 PLS2 中预测变量多于响应变量）

从而将 $ Q $ 的计算复杂度从 $ O(KM) $ 降至 $ O(M) $，实现 **$ O(K) $ 倍加速**。

---

### 相比现有方法的优势

| 方面 | 优势 |
|------|------|
| **正确性** | 改进后算法生成完全相同的 $ W, P, Q, R, T $，是“drop-in replacement” |
| **速度** | 在典型设置下，整体拟合提速达 **2×（CPU）至 6×（GPU）**；局部步骤提速可达 **两个数量级** |
| **适用性** | 所有 $ A \geq 2 $ 场景均可应用 $ R $ 的改进；大多数实际应用场景（如 PLS1 或 $ M < K $）可受益于 $ Q $ 的改进 |
| **实现友好** | 已集成到开源 Python 包 `ikpls`，支持 NumPy（CPU）与 JAX（GPU） |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
未使用真实世界数据集进行模型性能比较，而是采用**合成随机数据**进行基准测试（benchmarking），以控制变量、系统性地分析算法运行时间随维度变化的趋势。

参数范围覆盖：
- $ N \in \{10^3, 2\times10^2\} $：样本数
- $ K \in \{10^2, 10^3\} $：预测变量数（X 的列数）
- $ M \in \{1, 10, 10^3\} $：响应变量数（Y 的列数）
- $ A = 30 $：PLS 成分数固定为 30

---

### 实验设置和评估指标

#### 硬件平台
- **CPU**: AMD Ryzen 9 5950X（16核32线程），使用 NumPy 实现
- **GPU**: NVIDIA GeForce RTX 3090 Ti，使用 JAX 实现（JIT 编译）

#### 软件环境
- Python 3.14
- NumPy v2.5.1, JAX v0.10.2（CUDA 12.9）
- `ikpls` v6.1.2（作者自研开源包）

#### 评估指标
- **Wall-clock runtime**（壁钟时间）
- **Speedup ratio**（加速比）：原始版本 / 改进版本
- **Median of 10–1000 runs**，消除缓存冷启动等噪声影响
- 分别测量：
  - 单独 $ R $ 和 $ Q $ 步骤的时间
  - 完整 IKPLS 拟合时间

---

### 基线方法对比
- **Baseline**: 原始 IKPLS 算法（Dayal & MacGregor, 1997）实现
- **Proposed**: 本文提出的改进版 IKPLS（使用 Equation (2) 计算 $ R $，Algorithm 3 计算 $ Q $）
- 对比对象包括 IKPLS Algorithm 1 和 Algorithm 2 的原始与改进版本

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）单独步骤加速效果（图1 & 图2）

| 场景 | $ R $ 加速比 | $ Q $ 加速比 |
|------|--------------|---------------|
| CPU ($ K=10^3, A=30 $) | ~5–10× | 最高 **~100×**（$ M=1 $ 时） |
| GPU ($ K=10^3, A=30 $) | 最高 **~50×** | 最高 **~1000×**（$ M=1 $ 时） |

> 💡 特别地，当 $ M=1 $（PLS1）时，$ Q $ 的计算时间几乎恒定（约 8μs CPU / 100μs GPU），不再随 $ K $ 增长。

#### （2）完整拟合加速效果（图3）

| 设置 | CPU 加速比 | GPU 加速比 |
|------|------------|------------|
| $ N=10^3, K=10^3, M=1 $ | ~1.6× | ~4.7× |
| $ N=10^3, K=10^3, M=10 $（$ M < K $） | ~1.25× | ~1.24× |
| $ N=10^3, K=10^3, M=10^3 $（$ M \geq K $） | ~1.0×（无加速） | ~1.0×（无加速） |

> ✅ 当 $ M \geq K $ 时，$ Q $ 改进不生效，因此无额外收益。

---

### 与基线方法的对比结果
- 所有测试配置中，**改进版本均快于原始版本**
- 在常见化学计量学任务（如 NIR 光谱回归，通常 $ M=1 $）中，**GPU 上整体加速达 6×**
- $ R $ 的改进在所有 $ A \geq 2 $ 场景下都有效，且对总耗时贡献更大（尤其在高 $ A $ 时）

---

### 消融实验结果（隐含在分析中）
虽然没有显式命名“ablation study”，但通过分步计时实现了类似功能：
- 单独测试 $ R $ 和 $ Q $ 的运行时间 → 验证各自独立加速效果
- 不同 $ M/K $ 组合下的表现差异 → 验证 $ Q $ 改进的适用边界
- 发现：**$ R $ 的改进主要得益于更好的并行化**（相同乘法次数，更优执行模式）
- 发现：**$ Q $ 的改进真正降低了渐近复杂度**（从 $ O(KM) \to O(M) $）

---

## 4. 关键结论和发现

### 论文的主要发现

1. ✅ **数学等价性成立**：  
   新的 $ R $ 和 $ Q $ 计算方式与原算法**完全等价**，输出一致（Theorem 1, Corollary 1）。

2. ✅ **$ Q $ 存在长期被忽视的代数关系**：  
   首次严格证明 $ \mathbf{q}_a \propto (X^\top Y)_a \mathbf{w}_a $，解释了 R 包 `pls` 中注释 “is q proportional to q.a?” 的合理性，并给出比例常数。

3. ⚡ **实际加速显著**：
   - 局部步骤最高提速 **两个数量级**
   - 整体拟合在典型设置下提速 **2×（CPU）至 6×（GPU）**

4. 📈 **硬件越并行，收益越大**：  
   GPU 上的加速比远高于 CPU，说明新方法更适配现代加速器架构。

5. 🔁 **改进可无缝集成**：  
   所有改动均为局部替换，已在开源库 `ikpls` 中实现，用户无需修改调用代码即可受益。

---

### 方法的局限性

| 局限 | 说明 |
|------|------|
| $ Q $ 改进仅适用于 $ M=1 $ 或 $ M < K $ | 当 $ M \geq K $ 时退化为原始算法，无加速 |
| 不改变整体 PLS 渐近复杂度 | 总体仍为 $ O(NKA) $，仅优化其中部分步骤 |
| 依赖良好条件的数据 | 极端病态数据可能影响数值稳定性（虽原文称原 IKPLS 已稳定） |

---

### 未来工作方向

1. **扩展至其他 PLS 变体**：  
   将此类代数简化思想应用于 OPLS、Sparse PLS、Robust PLS 等。

2. **结合 fast cross-validation**：  
   作者已在 `ikpls` 中集成其先前工作 [Engstrom & Jensen, 2025] 的快速交叉验证算法，未来可研究联合优化策略。

3. **自动选择最优路径**：  
   在运行时根据 $ M, K $ 自动切换高效分支，最大化性能。

4. **探索更多隐式代数结构**：  
   是否还有其他中间变量可被复用？例如 $ T, W $ 的更新过程是否存在类似规律？

---

> 📦 **开源实现**：所有改进均已发布于免费开源 Python 包 [`ikpls`](https://github.com/olchark/ikpls)，支持 CPU/GPU，可通过 pip 安装使用。

</details>

---

### 16. [VarRate: Training-Free Variable-Rate KV Cache Compression for Long-Context LLMs](https://arxiv.org/abs/2607.15498)

**Authors**: Shahrzad Esmat, Dhawal Shah, Ali Jannesari  
**Category**: cs.CL  
**Published**: 2026-07-20  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.15498v1  

#### Abstract
The key-value (KV) cache is the main memory bottleneck in long-context large language model (LLM) inference. Two leading training-free families are both structurally limited: token-selection methods (SnapKV, Ada-KV) score importance from an observation window and evict low-scoring tokens, but evicti...

---

### 17. [SkillCorpus: Consolidating and Evaluating the Open Skill Ecosystem for Real-World LLM Agents](https://arxiv.org/abs/2607.15557)

**Authors**: Yanze Wang, Pengfei Yao, Tianyi Sun, Chuanrui Hu, Yan Xiao, Yunyun Han, Jun Sun, Yafeng Deng  
**Category**: cs.CL  
**Published**: 2026-07-20  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.15557v1  

#### Abstract
Agent skills, SKILL.md files that package reusable procedural knowledge for an LLM agent, are a popular mechanism for extending agent capabilities. Public repositories now host them in large and growing numbers, yet these artifacts are fragmented, redundant, and uneven in quality, and their value in...

---

### 18. [Data-Native Global Optimization for Big Data K-means Clustering](https://arxiv.org/abs/2607.15835)

**Authors**: Ravil Mussabayev, Rustam Mussabayev, Zukhra Yerdaliyeva, Kuldeyev Nursultan  
**Category**: cs.LG  
**Published**: 2026-07-20  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.15835v1  

#### Abstract
Big data clustering remains challenging: the Minimum Sum-of-Squares Clustering (MSSC) problem underlying K-means is NP-hard, and existing methods either reach poor local minima or require prohibitive metaheuristic hybrids. We target arbitrarily tall data: a fixed feature space may contain arbitraril...

---

### 19. [FSZ: Breaking the Prediction-Throughput Trade-off in GPU Lossy Compression](https://arxiv.org/abs/2607.15413)

**Authors**: Jiajun Huang  
**Category**: cs.DC  
**Published**: 2026-07-20  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.15413v1  

#### Abstract
Existing fast GPU error-bounded lossy compressors have achieved high throughput through pure-GPU single-kernel designs, but their compression ratios remain limited because they typically apply a fixed first-order predictor on independent blocks. We propose FSZ, a GPU error-bounded lossy compressor t...

---

### 20. [A Transportable Threshold-Based Framework for Interpretable Classification of Medical Data](https://arxiv.org/abs/2607.15394)

**Authors**: Antony Garcia, Adrian Noriega, Gabrielle Britton, Xinming Huang  
**Category**: cs.LG  
**Published**: 2026-07-20  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.15394v1  

#### Abstract
Black-box models limit the adoption of artificial intelligence in medicine due to their lack of interpretability and reproducibility. We introduce a statistically grounded framework that provides fully interpretable, rule-based clinical classification using the Bernoulli Na\"ive Bayes (BNB) model. T...

---

### 21. [Graph Coloring Approach to Solving Sudoku with Oscillatory Neural Networks](https://arxiv.org/abs/2607.15814)

**Authors**: Filip Sabo, Aida Todri-Sanial  
**Category**: cs.LG  
**Published**: 2026-07-20  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.15814v1  

#### Abstract
Oscillatory Neural Networks (ONNs) present an attractive physics-based computing paradigm rooted in the dynamics of a network of typically fully coupled oscillators aiming to minimize an underlying energy function. In this paper, we propose an ONN-based solver for one well-known constrained combinat...

---

### 22. [Causal-Audit: Explicit and Auditable Graph-based Reasoning via Target-Aware Causal Chain Construction](https://arxiv.org/abs/2607.15281)

**Authors**: Su Lan, Xuefei Yin, Yanming Zhu, Alan Wee-Chung Liew  
**Category**: cs.AI  
**Published**: 2026-07-20  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.15281v1  

#### Abstract
Causal and intervention-based question answering is fundamental to advancing large language models (LLMs) toward reasoning beyond surface-level correlations and understanding underlying causal mechanisms. However, existing LLM-based methods often rely on implicit language-level reasoning, resulting ...

---

### 23. [ToolVerse: Unlocking Massive Environments and Long-Horizon Tasks for Agentic Reinforcement Learning](https://arxiv.org/abs/2607.15660)

**Authors**: Shuaiyu Zhou, Fengpeng Yue, Zengjie Hu, Yuanzhe Shen, Chenyang Zhang, feng hong, Cao Liu, Ke Zeng  
**Category**: cs.AI  
**Published**: 2026-07-20  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.15660v1  

#### Abstract
While LLM agents demonstrate strong reasoning abilities in compact and well-defined scenarios, they struggle to maintain robustness and effectiveness when faced with large-scale, diverse, and dynamic real-world environments that demand seamless tool integration. To address this gap, we introduce Too...

---

### 24. [S1-Omni: A Unified Multimodal Reasoning Model for Scientific Understanding, Prediction, and Generation](https://arxiv.org/abs/2607.15686)

**Authors**: Jiahao Zhao, Junyi Liu, Lifeng Xu, Nan Xu, Qingli Wang, Qingxiao Li, Tianle Chen, Xiaoyu Wu, Yawen Zheng, Zikai Wang, Guanming Liu, Hequn Zhou, Jingyi Wang, Jingyuan Shu, Keqi Wang, Li He, Songyang Diao, Wenhui Xu, Xinyu Ren, Yaqin Fan, Yujin Zhou, Zhanao Yao  
**Category**: cs.AI  
**Published**: 2026-07-20  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.15686v1  

#### Abstract
We present S1-Omni, a unified multimodal reasoning model for scientific understanding, prediction, and generation. AI for Science (AI4S) has advanced significantly through domain-specific models, tool-augmented LLMs, and scientific language models. However, model capabilities remain highly fragmente...

---

### 25. [NeurOWL: An LLM-Based Neural-symbolic Framework for Incomplete OWL Ontology Reasoning](https://arxiv.org/abs/2607.15776)

**Authors**: Hui Yang, Jiaoyan Chen, Yiping Song, Renate Schmidt, Wen Zhang  
**Category**: cs.AI  
**Published**: 2026-07-20  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.15776v1  

#### Abstract
OWL ontologies provide a formal knowledge representation framework that enables semantic reasoning, and have been widely adopted across domains such as healthcare and bioinformatics. In practice, however, real-world ontologies are often incomplete, which pose challenges for reasoning. In this work, ...

---

### 26. [Knowledge-Centric Agents for Workflow Generation](https://arxiv.org/abs/2607.15845)

**Authors**: Zhendong Li, Lei Sun, Ruibo Ming, He Zhang, Danda Pani Paudel, Luc Van Gool, Jinjin Gu  
**Category**: cs.AI  
**Published**: 2026-07-20  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.15845v1  

#### Abstract
Workflow generation in visual creation systems such as ComfyUI demands not only syntactic accuracy but also expert-level reasoning over modular compositions. Existing large language model (LLM) approaches often treat this as a direct text-to-JSON generation task, struggling with structural brittlene...

---

### 27. [Behaviour-Conditioned Neural Processes for Adaptive Residential Short-Term Load Forecasting](https://arxiv.org/abs/2607.16168)

**Authors**: Ramin Soleimani, Andrea Visentin, Dirk Pesch  
**Category**: cs.LG  
**Published**: 2026-07-20  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.16168v1  

#### Abstract
Residential short-term load forecasting (STLF) is challenging because household demand is heterogeneous, temporally variable, and shaped by diverse behavioural routines. This work investigates whether inferred behavioural structure can be embedded within the forecasting mechanism of a Neural Process...

---

### 28. [Physics-enhanced reinforcement learning for real-time optimal control of dynamical systems](https://arxiv.org/abs/2607.16177)

**Authors**: Matteo Tomasetto, Nicol\`o Botteghi, Gabriele Bruni, Andrea Manzoni  
**Category**: cs.LG  
**Published**: 2026-07-20  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.16177v1  

#### Abstract
Reinforcement learning (RL) has recently emerged as a promising feedback control strategy for nonlinear and complex dynamical systems. However, RL algorithms are sample inefficient and require a large number of interaction with the environment to synthesize optimal control strategies. Consequently, ...

---

### 29. [A Blueprint for Equilibrium-Based Differentiable Continuous-Variable Thermodynamic Computing](https://arxiv.org/abs/2607.16183)

**Authors**: Owen Lockwood, J\'er\'emy B\'ejanin, Joost Bus, Christopher Chamberland, Patrick Huembeli, Frank Sch\"afer, Guillaume Verdon  
**Category**: cs.LG  
**Published**: 2026-07-20  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.16183v1  

#### Abstract
To address the escalating energy and latency demands of machine-learning workloads, we introduce a blueprint for an energy-efficient and fast thermodynamic computing stack that leverages stochastic analog processes in physical hardware. In this work, we focus on energy-based thermodynamic computing ...

---

### 30. [GraphDx: A Cost-Aware Knowledge-Enhanced Multi-Agent Framework for Sequential Diagnosis](https://arxiv.org/abs/2607.15280)

**Authors**: Shaoting Tan, Ning Liu, Yuntao Du, Shuyue Wei, Wu Shuai, Qian Li, Yanyu Xu, Wei Zhang, Lizhen Cui, Haitao Yuan  
**Category**: cs.AI  
**Published**: 2026-07-20  
**Score**: 3.5  
**Type**: new  
**ArXiv ID**: 2607.15280v1  

#### Abstract
Sequential diagnosis requires balancing diagnostic accuracy against resource costs through iterative information gathering. Existing Large Language Model (LLM) approaches exhibit a critical knowledge-reasoning gap: despite encoding extensive medical knowledge, they struggle to reason systematically ...

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

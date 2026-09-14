# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-09-14 11:02:36 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Attention Quantization for Tabular Foundation Models](https://arxiv.org/abs/2609.13031)

**Authors**: Jonas M. K\"ubler, Benjamin J\"ager, Klemens Fl\"oge, Noah Hollmann, Frank Hutter  
**Category**: cs.LG  
**Published**: 2026-09-14  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2609.13031v1  

#### Abstract
With the recent rise and adoption of tabular foundation models, optimizing their inference performance becomes an emerging field for efficiency research. While the models are architecturally similar to transformer-based large language models (LLMs), the size and serving patterns differ significantly...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Attention Quantization for Tabular Foundation Models**

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
随着 **tabular foundation models**（如 TabPFN、TabICL）在表格数据上的广泛应用，其推理效率成为实际部署中的瓶颈。尽管这些模型架构上类似于 **LLM** 中的 Transformer，但由于其独特的 **in-context learning (ICL)** 范式，计算成本随训练样本数 $N$ 呈 **$O(N^2 + MN)$** 增长（$M$ 为测试样本数），导致大规模数据下推理延迟显著。

传统针对 LLM 的优化方法（如权重量化、KV Cache 量化）在 tabular 模型中收益有限，因为：
- 模型本身较小，权重量化节省有限；
- 主要开销来自注意力机制中对大量训练行的处理。

本文提出应将优化重点从“权重”转向“注意力计算”，特别是 **query、key、value 的矩阵乘法运算**。

---

### **提出了什么新方法或新思路**
作者提出了一种 **FP8 注意力量化策略**，核心创新如下：

- **FP8 量化注意力输入**：将 ICL 层中的 queries ($Q$)、keys ($K$)、values ($V$) 动态量化至 **e4m3fn FP8** 格式，并利用现代 GPU（如 NVIDIA Hopper/Blackwell）支持的 **FP8 Tensor Core** 加速矩阵乘法。
- **Per-head AbsMax 缩放**：每个 attention head 单独计算量化 scale，以最大化利用 FP8 表示范围。
- **Train-Test Quantization Coordination**：关键发现是必须**同时且一致地量化 train-train 和 test-train 注意力路径**，否则精度会大幅下降。为此，作者复用训练 query 的量化 scale 到测试 query 上，确保误差对齐。
- **Gating 机制**：仅当训练样本数 $N > 8192$ 时启用 FP8 内核，避免小数据下的量化开销超过收益。

该方法通过自定义 **Triton kernel** 实现，紧密参考 FlashAttention-2 设计，并集成 FP8 MMA 指令。

---

### **相比现有方法的优势**
| 维度 | 本文方法 | 现有主流方法（如 LLM 量化） |
|------|----------|-----------------------------|
| 优化目标 | 注意力计算（主导瓶颈） | 权重 / KV Cache |
| 适用场景 | 大规模 tabular 数据（$N \gg 1$） | 大语言模型生成任务 |
| 硬件利用率 | 充分利用 FP8 Tensor Core 吞吐优势 | 多基于 INT8/BF16 |
| 精度保持 | 几乎无损（误差 < 种子噪声） | 可能引入显著退化 |
| 推理加速 | 最高达 **1.7x end-to-end speedup** | 在 tabular 场景加速有限 |

> ✅ **核心优势**：首次系统研究并实现 **FP8 attention quantization for tabular foundation models**，实现了**高吞吐、低延迟、几乎无损**的推理加速。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **TabArena**：包含 51 个表格数据集的基准，用于评估模型质量（Elo 分数、metric error）。
- **BeyondArena**：更大更复杂的基准，包含 142 个数据集，其中 56 个超过 8192 行，部分达到百万级样本，用于验证可扩展性和端到端性能。

---

### **实验设置和评估指标**

#### **模型**
- 主要实验基于 **TabPFN-v3**（Grouped Query Attention 架构）
- 验证泛化性时使用 **TabICLv2**

#### **硬件平台**
- 主要运行于 **NVIDIA RTX Pro 6000 Blackwell Edition**
- 补充实验在 **L4 GPU** 和不同 head dimension（64 vs 128）上进行

#### **评估指标**
| 类别 | 指标 |
|------|------|
| **准确性** | Elo score、relative metric error（roc_auc/log_loss/rmse 混合）、sign test p-value |
| **性能** | attention kernel 延迟（ms）、end-to-end predict 时间（s）、speedup 倍数 |
| **鲁棒性** | 跨 3 个随机种子的结果波动 |

#### **基线方法对比**
- **Baseline**：`torch.nn.functional.scaled_dot_product_attention`（调用 FlashAttention-2）
- **Control**：自研 16-bit Triton kernel（排除设计优化影响）
- **Variants**：
  - 不同量化范围（仅 train-train / 仅 test-train / both）
  - 是否共享 query scale
  - 是否启用 $N > 8192$ 门控（gating）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **速度提升（TabPFN-v3, RTX Pro 6000）**
| $N$（训练行数） | Kernel Speedup | End-to-End Speedup |
|------------------|----------------|--------------------|
| 16k              | 1.46×          | 1.08×              |
| 64k              | 1.66×          | 1.37×              |
| 256k             | 1.71×          | 1.61×              |
| 512k             | **1.72×**      | **1.67×**          |

> 🔹 量化开销在 $N > 8192$ 后被摊销，FP8 开始带来正向收益。

#### **TabICLv2 上的加速效果**
| $N$（训练行数） | End-to-End Speedup |
|------------------|--------------------|
| 512k             | **1.56×**          |

> 尽管绝对加速略低于 TabPFN-v3，但仍显著，说明方法具有跨模型通用性。

#### **不同硬件与配置表现**
- 在 **L4 GPU** 上最高达 **1.89× kernel speedup**
- 使用 **head dim=128** 时，FP8 speedup 达 **1.82×**，表明 head 越大，矩阵乘法占比越高，FP8 收益越大

---

### **与基线方法的对比结果**

| 方面 | 结果 |
|------|------|
| **准确性（TabArena）** | FP8 版本 Elo 差异在种子噪声范围内：<br>Baseline: `1648.7±2.1` → FP8: `1648.4±2.1` |
| **准确性（BeyondArena）** | 平均 Elo 变化 `-2.0` 至 `+0.6`，统计不显著（p > 0.05） |
| **相对误差变化** | 平均 △err < 0.03%，远小于预处理种子差异（约5倍） |
| **batch invariance** | 使用共享 query scale 后，预测结果不再依赖 test batch size |

> ✅ **结论**：FP8 attention 在正确配置下**几乎无损精度**，且可保证部署一致性。

---

### **消融实验结果**

#### **表1：不同量化策略的质量影响（TabArena）**
| 量化范围 | △error (%) | △Elo | Sign Test p |
|---------|------------|-------|-------------|
| train→train only | +2.81 | -31.8 | <0.001 |
| test→train only | +2.38 | -20.3 | <0.001 |
| both, per-call scale | +0.01 | -1.0 | 0.024–0.131 |
| **both, shared train scale** | **+0.02** | **-1.3** | **0.024–0.080** |

> ⚠️ 单独量化任一路径都会造成显著退化；只有**两者同时量化且 scale 对齐**才能接近无损。

#### **图3：Gaussian 噪声实验证明“协调性”必要**
- 添加独立噪声到两个 attention 路径 → 性能急剧下降
- 若 K/V 的噪声在两条路径间“coordinated”（即相同）→ 模型鲁棒性强得多
> 👉 证明模型内部存在一种**隐式的误差容忍机制**，前提是 train/test 输入扰动方式一致。

#### **门控（gating）有效性**
- $N < 8192$ 时不开启 FP8，避免负优化
- 图1显示：超过阈值后 wall-clock time 显著降低（BeyondArena 下降超 70 分钟）

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **注意力计算是 tabular foundation models 的主要瓶颈**，应优先优化而非权重。
2. ✅ **FP8 attention quantization 可实现高达 1.7x 的 end-to-end 加速**，尤其适用于大数据集（$N > 8k$）。
3. ✅ **必须同时量化 train-train 和 test-train 注意力路径**，并保持量化参数一致（如 scale），否则精度严重受损。
4. ✅ **复用训练 query 的量化 scale 到测试 query 上**，可在保持精度的同时实现 batch-invariant 推理。
5. ✅ 所引入的误差**小于 TabPFN 自身预处理的种子方差**，因此在实践中可忽略。

> 🌟 **最反直觉但最重要的发现**：虽然分别加噪看似更保守，但**统一扰动反而更鲁棒**——这揭示了 tabular foundation models 对“上下文一致性”的深层依赖，是 LLM 中未见的现象。

---

### **方法的局限性**
1. **硬件依赖性强**：需要支持 FP8 Tensor Core 的设备（如 Blackwell/Hopper），不适用于 H100/B200 以外的老架构。
2. **Softmax 可能成新瓶颈**：在某些高端芯片（如 B200）上，即使加速了 MatMul，softmax 仍可能成为瓶颈（Zadouri et al., 2026）。
3. **未使用 TMA 等高级特性**：当前 kernel 基于 FA2 架构，未利用 TMA（Tensor Memory Accelerator）等最新硬件功能，在部分平台上无法达到最优性能。
4. **仅限推理阶段**：为 post-training 优化，不影响训练过程。

---

### **未来工作方向**
1. **适配更多硬件平台**：开发支持 TMA 的 FP8 attention kernel，适配 H100/B200 等主流 AI 加速器。
2. **探索其他低精度格式**：如 INT8 或混合精度方案，进一步降低成本。
3. **扩展至其他 tabular 架构**：验证在非 Transformer 架构中的可行性。
4. **结合稀疏注意力**：在极大规模数据（$N > 1M$）下结合 sparse attention 进一步降低复杂度。
5. **理论解释“coordination effect”**：为何模型能容忍 coordinated 扰动而不能容忍 uncoordinated？是否反映某种不变性？

---

> 💡 **总体评价**：本文为 **tabular foundation models 的高效推理开辟了新路径**，强调了领域特定优化的重要性，并揭示了一个新颖的“量化协调性”原则，对后续研究具有重要指导意义。

</details>

---

### 2. [AMDKernelVault: Large-Scale Datasets and Agentic Training for AMD GPU Kernel Optimization](https://arxiv.org/abs/2609.12471)

**Authors**: Ji Liu, Saptarshi Majumder, Yiqing Huang, Wenwen Ouyang, Umang Pandey, Zeping Li, Chushi Chen, Zihao An, Puyuan Yang, Zekai Li, Sina Rafati, Ziqiong Liu, Pratik Prabhanjan Brahma, Dong Li, Zicheng Liu, Sharon Zhou, Emad Barsoum  
**Category**: cs.CL  
**Published**: 2026-09-14  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.12471v1  

#### Abstract
We introduce AMDKernelVault, an open HIP and Triton kernel corpus and training framework for recent AMD CDNA GPUs. Existing LLM-based kernel agents are largely CUDA/NVIDIA-centric and often depend on repeated frontier-LLM calls for generation, reflection, and optimization. To address this gap, we de...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# AMDKernelVault 论文核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

当前基于 LLM 的 GPU 内核优化系统（如 CUDA/NVIDIA）高度依赖前沿大模型（frontier LLMs），且生成流程多为 **CUDA-centric**，缺乏对 **AMD GPU** 和其原生编程栈（HIP/ROCm）的支持。具体问题包括：

- **数据缺失**：缺少大规模、经过执行验证的 AMD HIP 和 Triton 内核数据集。
- **移植不可靠**：直接通过 `hipify` 工具将 CUDA 转换为 HIP 可能保留 NVIDIA 特定假设（如 warp size、tiling 策略），导致在 AMD 上编译失败或性能不佳。
- **部署成本高**：现有 agent 流程反复调用云端大模型（如 GPT-5），增加延迟与成本，难以在私有 AMD 环境中本地化部署。

---

### 🚀 提出的新方法与创新点

#### （1）**AMDKernelVault：首个面向 AMD 的大规模执行验证内核语料库**

- 包含：
  - **62,153 个执行验证的 HIP 内核样本**
  - **39,893 个 Triton 内核样本**
  - **2,377 条来自 rocBLAS/rocSOLVER 的生产级 QA 监督数据**
- 所有内核均在真实 AMD CDNA GPU 上完成：
  - 编译（via ROCm）
  - 正确性验证（numerical equivalence）
  - 延迟分析（latency profiling）

#### （2）**两套可扩展的数据生成管道：HIPKernelGen 与 TritonKernelGen**

- 基于 **generate-evaluate-reflect** 的 agent 范式构建：
  - 输入：PyTorch 函数式参考实现
  - 输出：HIP 或 Triton 内核代码
  - 验证闭环：编译 → 运行时正确性检查 → 性能反馈 → 失败则触发反思重生成
- 支持多种来源输入：
  - 合成任务（CUDA-Agent-Ops-6K）
  - GitHub 衍生模块（GPUMODE-KernelBook）
  - 生产库代码（rocBLAS/rocSOLVER）

#### （3）**训练紧凑型本地 LLM 用于 AMD 内核生成**

- 使用 **Qwen3-8B** 模型进行两阶段训练：
  1. **监督微调（SFT）**：学习 HIP/Triton 语法与常见模式
  2. **执行感知强化学习（Execution-aware RL）**：利用编译、正确性和速度信号作为奖励
- 单一策略模型担任 **generator、reflector、optimizer** 三重角色，在 GEAK-style agent loop 中运行

---

### 🔍 相比现有方法的优势

| 维度 | 传统方法（如 Kevin, GEAK） | AMDKernelVault |
|------|-----------------------------|----------------|
| 平台支持 | 主要针对 CUDA/NVIDIA | 原生支持 AMD HIP/ROCm/Triton |
| 数据质量 | 多为语法转换或未验证代码 | 全部经过编译+执行+性能验证 |
| 部署方式 | 依赖外部 frontier LLM 调用 | 可训练本地 8B 模型，降低延迟与成本 |
| 训练目标 | 多数仅做生成 | 支持端到端 agent 角色内部化 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 数据源 | 类型 | 数量 | 描述 |
|-------|------|------|------|
| **HIP-CudaAgent** | PyTorch → HIP | 5,388 | 来自 CUDA-Agent-Ops-6K，经 HIPKernelGen 验证 |
| **HIP-GPUMode** | PyTorch → HIP | 22,397 | 来自 GPUMODE-KernelBook，平均每个任务 3.8 个变体 |
| **HIP2HIP** | HIP → HIP 优化 | 34,368 | 对已有 HIP 内核进行优化 |
| **ROCm Libraries QA** | 生产级监督 | 2,377 | 来自 rocBLAS/rocSOLVER 的接口级 QA 对 |
| **Triton-Stack** | Web 抓取 + 过滤 | 2,269 | 中等难度 Triton-PyTorch 对 |
| **Triton-Bench** | Benchmark 衍生 | 7,713 | 来自 TritonBench-8k，去 DSL 化处理 |
| **Triton-GPUMode** | Torch Inductor 导出 | 18,000 | 支持通用形状参数 |
| **Triton-AICE** | AI-CUDA-Engineer 转换 | 11,911 | 将 CUDA Triton 转为 AMD 兼容版本 |

> ✅ 总计：**64,530 HIP/ROCm 样本**，**39,893 Triton 样本**

---

### ⚙️ 实验设置与评估指标

#### 评估任务

| 任务 | 输入 | 输出 |
|------|------|------|
| **PyTorch → HIP** | PyTorch 模块 | 功能等价 HIP 内核 |
| **HIP → HIP** | 基线 HIP 内核 | 更快的优化版本 |
| **Text → Triton** | 自然语言描述 | Triton 内核 |
| **Triton → Triton** | 原始 Triton 内核 | 优化版（tiling, memory 等） |

#### 评估指标

| 指标 | 定义 |
|------|------|
| **Comp@k** | 第 k 次迭代前成功编译的比例 |
| **Corr@k** | 第 k 次迭代前数值正确的比例 |
| **Pass@k** | k 次独立采样中至少有一个正确的概率 |
| **Speed@10** | 最佳 10 个正确样本的平均加速比（vs PyTorch 或输入内核） |

#### 基线对比模型

- **GPT-5**
- **Gemini 2.5 Pro**
- **Claude Sonnet 4**
- **Qwen3-8B（base）**
- **Qwen3-8B + SFT**
- **Qwen3-8B + SFT + RL**

所有模型在同一 agent 框架（GEAK OptimAgent-v2）下测试，固定预算：`max_iteration=3`, `num_offsprings=1`

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（固定预算下）

#### 表 1：PyTorch → HIP 翻译（200 个任务）

| Model | Comp Acc | **Corr Acc (Pass@1)** |
|-------|----------|------------------------|
| GPT-5 | 75.5% | 24.0% |
| Qwen3-8B (base) | 50.0% | 14.5% |
| Qwen3-8B + SFT | 78.0% | 31.5% |
| **Qwen3-8B + SFT + RL** | **83.5%** | **34.0%** ✅ |

> ✔️ 在 **正确率上超越所有 frontier LLM**

---

#### 表 2：HIP → HIP 优化（100 个任务）

| Model | Corr Acc | Speed@10 |
|-------|----------|-----------|
| GPT-5 | 87% | 1.32× |
| Qwen3-8B + SFT + RL | **84%** | **1.14×** |

> ✔️ 接近 GPT-5 正确率，具备正向加速能力

---

#### 表 3：TritonBench-G（184 kernels）

| Model | Corr@1 | **Corr@3** | Comp@3 | Speed |
|-------|--------|------------|--------|-------|
| GPT-5 | 3.8% | 15.2% | 28.8% | 1.05× |
| Claude Sonnet 4 | 7.6% | 29.8% | 46.7% | 1.34× |
| **Qwen3-8B + SFT + RL** | **12.0%** | **33.2%** ✅ | **67.9%** ✅ | **1.46×** |

> ✔️ 在 **Corr@3** 和 **编译成功率** 上全面领先

---

#### 表 4：ROCmBench（31 个 AMD 特定内核）

| Model | Corr@1 | **Corr@3** | Comp@3 | Speed |
|-------|--------|------------|--------|-------|
| GPT-5 | 16.13% | 29.03% | 35.48% | 1.31× |
| Claude Sonnet 4 | 19.35% | 35.48% | **67.74%** | **1.82×** |
| **Qwen3-8B + SFT + RL** | **22.58%** | **41.94%** ✅ | 58.06% | 1.61× |

> ✔️ **Corr@3 达到最高 41.94%**，显著优于其他模型  
> ❗ 但在编译率和最高速度上未完全超越 Claude

---

### 🔍 消融实验结果（Ablation Studies）

#### （1）数据规模影响（PyTorch → HIP）

| 数据比例 | Corr Acc |
|---------|----------|
| 0% (base) | 14.5% |
| 10% | 17.5% |
| 50% | 23.0% |
| **100%** | **34.0%** |

> ➕ 数据越多，性能越强，证明语料库高质量有效

#### （2）训练组件消融（TritonBench-G）

| 配置 | Corr@3 | Δ |
|------|--------|----|
| Full system (SFT + RL) | **33.2%** | — |
| Single-turn (T=1) | 20.1% | -13.1% |
| No SFT cold start | 8.5% | -24.7% |

> ✅ **SFT 是关键冷启动步骤**  
> ✅ **多轮反思（multi-turn reflection）带来巨大增益**

#### （3）固定 agent 框架下的模型作用

| 模型 | Corr@3 |
|------|--------|
| Qwen3-8B (base) | 4.9% |
| + SFT | 10.3% |
| + SFT + Agentic RL | **33.2%** |

> ✔️ 仅靠外部 agent 框架无法提升性能，必须对模型进行 **agent-style 反馈训练**

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **AMDKernelVault 是首个面向 AMD 的执行验证内核语料库**，填补了 ROCm 生态在数据层面的空白。
2. 利用该语料库训练的 **Qwen3-8B 模型**，在相同评估预算下，在多个关键指标（如 Pass@1、Corr@3）上 **超过 GPT-5、Claude 等 frontier LLM**。
3. **SFT + Execution-aware RL** 的训练范式能有效教会小模型理解编译错误、数值偏差和性能瓶颈，并实现自我修正。
4. **单一本地 8B 模型可承担 generator/reflector/optimizer 三重角色**，有望替代昂贵的云端 LLM 调用链，推动本地化 agent 部署。

---

### ⚠️ 局限性

1. **专家级内核仍难生成**：D5 级别（expert）任务所有模型均为 0/5，表明复杂多阶段 reduction、wavefront 同步等问题仍未解决。
2. **部分指标未全面领先**：虽然正确率最优，但在编译率和峰值速度上尚未超越最强 baseline（如 Claude）。
3. **潜在结构继承问题**：部分生成内核可能继承自 CUDA 源头的次优结构（如固定 block size）。
4. **缺乏统一语义去重审计**：未进行全库级语义重复检测，可能存在功能近似样本。
5. **环境版本不一致**：不同数据生成阶段使用的 ROCm/PyTorch 版本略有差异。

---

### 🔮 未来工作方向

1. **引入更丰富的性能反馈机制**：加入 profiler-guided feedback（如 memory footprint、register pressure、occupancy）以指导更深层次优化。
2. **扩大生产级监督覆盖范围**：从更多 rocLibraries（如 rocSPARSE、MIOpen）提取接口级 QA 数据。
3. **构建 per-operator-family 分析体系**：按 GEMM、Softmax、Normalization 等分类报告 pass rate 与 speedup。
4. **探索跨平台迁移能力**：研究在 MI325 上训练的优化是否适用于 MI250 或未来 CDNA 架构。
5. **开发专用评估基准**：建立专门针对 AMD 的 KernelBench-AMD，避免对 CUDA 衍生任务的依赖。

---

> 🔗 **资源公开地址**：
> - 数据集：[https://huggingface.co/datasets/amd/AIG-Datasets](https://huggingface.co/datasets/amd/AIG-Datasets)
> - 代码仓库：[https://github.com/AMD-AGI/hip_kernel_llm_lab](https://github.com/AMD-AGI/hip_kernel_llm_lab)

</details>

---

### 3. [Expert-Space Exploration in MoE Reinforcement Learning](https://arxiv.org/abs/2609.13058)

**Authors**: Hongyi He, Zhenghao Lin, Xiao Liu, Peng Cheng, Yan Lu, Yeyun Gong  
**Category**: cs.CL  
**Published**: 2026-09-14  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.13058v1  

#### Abstract
Reinforcement learning (RL) has become central to post-training of large language models. Recent advances in RL for Mixture-of-Experts (MoE) models have primarily focused on improving optimization stability and training efficiency, while treating the expert selection as a fixed component. Since rout...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《Expert-Space Exploration in MoE Reinforcement Learning》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **Mixture-of-Experts (MoE)** 架构的大语言模型在进行 **Reinforcement Learning (RL)** 后训练时，通常将专家选择（expert routing）视为固定的架构组件，仅依赖 token-level 的采样（如 temperature sampling）来实现 rollout 多样性。这种做法导致以下问题：

- **探索空间受限**：确定性的 Top-K 路由机制使得相同前缀反复激活相同的稀疏计算路径，限制了潜在的多样化推理轨迹。
- **rollout 多样性下降**：随着 RL 训练推进，策略趋于集中，导致生成响应趋同，削弱了 group-relative 方法（如 GRPO）中的优势信号。
- **直接扰动质量差**：虽然对路由 logits 注入噪声可增加多样性，但无约束扰动容易激活不合适的专家，严重损害生成质量。

### 提出的新方法：ESRL
作者提出 **Expert-Space Exploration Reinforcement Learning (ESRL)**，一种架构感知的框架，显式地在 MoE 模型的 **expert-routing space** 中进行探索。

#### 核心思想
将 RL 中的探索维度从传统的 **token space** 扩展到 **computation-path space**，通过控制性扰动 expert routing 来生成多样化的 rollout，同时保持高质量输出。

#### 关键技术组件
1. **Entropy-Adaptive Noise Scaling**
   - 根据原始 router 分布的归一化熵动态调整噪声强度：
     $$
     \sigma_{t,l} = \sigma_{\min} + (\sigma_{\max} - \sigma_{\min})(1 - H_{t,l})
     $$
   - 高置信度（低熵）路由施加更强扰动以促进探索；低置信度（高熵）则减少扰动避免过度破坏。

2. **Anchored Expert Sampling**
   - 将 K 个被激活专家分为两部分：
     - **Anchored Experts**：保留 Top-K 中最可信的部分作为锚点，确保核心计算路径稳定。
     - **Exploratory Experts**：从一个候选池 $C_{\text{explore}}$ 中基于扰动后的 logits 随机选择其余专家。
   - 这种设计在保留可靠性的同时引入可控多样性。

3. **Routing Replay (R3)**
   - 在 rollout 阶段记录实际使用的 expert activation 路径。
   - 在 policy optimization 阶段重放这些路径，确保训练与生成阶段的 routing 行为一致，缓解 mismatch 问题。

### 相比现有方法的优势
| 方面 | 传统方法 | ESRL |
|------|--------|-------|
| 探索维度 | 仅 token-level sampling | 新增 expert-path level 探索 |
| 控制性 | 无控制或全局固定噪声 | 自适应噪声 + 锚定机制 |
| 一致性 | 可能存在 rollout-training mismatch | 使用 R3 保证路径一致性 |
| 正交性 | 多数修改 reward 或 loss | 完全在 rollout 阶段操作，与优化目标正交 |

> ✅ **核心优势**：ESRL 不改变 reward 函数或 policy objective，是一种可插拔的 rollout 增强模块，兼容并可叠加于其他 RL 改进方法之上。

---

## 2. 核心实验方法和设置

### 使用的数据集
#### 数学推理任务
- **OlympiadBench**：奥赛级别数学与物理问题
- **AIME 2024**：美国邀请数学考试
- **AMC**：美国数学竞赛
- **MinervaMath**：涵盖代数、微积分等高级数学领域

#### 科学推理任务
- **GPQA Diamond**：研究生水平的生物、物理、化学多选题
- **MMLU-Pro / MMLU-Redux**：更具挑战性的跨学科知识理解基准

#### 编程任务
- **LiveCodeBench v6**：基于执行测试用例评估代码生成能力，降低污染风险

---

### 实验设置
| 参数 | 设置 |
|------|------|
| 主干模型 | Qwen3-30B-A3B-Base / Instruct, Sigma-20B-A0.5B, Moonlight-16B-A3B |
| MoE 结构 | 支持 Top-K、Top-1 和含共享专家的不同变体 |
| RL 算法 | 基于 **GRPO**（Group Relative Policy Optimization） |
| Batch Size | 256 prompts × 8 responses → 2048 rollouts per step |
| 训练步数 | 300 iterations |
| 温度 | Rollout 使用 T=0.8, top-p=0.95 |
| 评估方式 | 独立采样 32 responses，报告 Pass@1 和 Pass@8 |

---

### 评估指标
- **Pass@1**：单次采样正确率（平均准确率）
- **Pass@8**：从 8 次采样中至少有一次正确的无偏估计
- **Self-BLEU**：衡量同一提示下多个响应之间的相似性，越低表示多样性越高
- **Informative Group Ratio**：包含正确与错误响应的组比例，反映 group-relative 学习信号强度
- **Expert Load CV / Imbalance Factor**：衡量专家利用率均衡性

---

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **GRPO** | 基础 RL 方法 |
| **GRPO-R3** | 加入 Routing Replay 提升稳定性 |
| **GSPO** | 序列级策略优化，提升 MoE 稳定性 |
| **Aux-Loss** | 添加负载均衡辅助损失 |
| **N-Sampling** | 对 router logits 进行采样 |
| **RO-GRPO** | 将路由统计信息纳入奖励机制 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（数学任务平均表现）

| 方法 | Avg. Pass@1 | Δ vs GRPO | Avg. Pass@8 | Δ vs GRPO |
|------|------------|----------|-------------|-----------|
| GRPO | 38.9 | — | 59.7 | — |
| GRPO-R3 | 39.7 | +0.8 | 61.1 | +1.4 |
| GSPO | 41.6 | +2.7 | 63.3 | +3.6 |
| **ESRL** | **42.1** | **+3.2** | **64.2** | **+4.5** |

> 🔥 在 **Qwen3-30B-A3B** 上，ESRL 显著优于所有基线，在 Pass@1 和 Pass@8 上分别提升 **3.2 和 4.5 个百分点**。

---

### 跨模型泛化能力
| 模型 | 方法 | Δ Pass@1 | Δ Pass@8 |
|------|------|---------|---------|
| Sigma-20B-A0.5B | ESRL vs GRPO | +0.8 | +2.7 |
| Moonlight-16B-A3B | ESRL vs GRPO | +1.3 | +3.5 |

✅ 表明 ESRL 在不同 MoE 结构（top-1、shared-expert）下均有效。

---

### 跨领域泛化（科学 & 编程）
在 **Qwen3-30B-A3B** 上进一步验证：

| 任务 | 指标 | GRPO | ESRL | Δ |
|------|------|------|------|----|
| GPQA | Pass@8 | 53.5 | **76.8** | **+23.3** |
| MMLU-Pro | Pass@8 | 67.4 | **82.5** | **+15.1** |
| MMLU-Redux | Pass@8 | 86.3 | **94.5** | **+8.2** |
| LiveCodeBench | Pass@8 | 52.2 | **54.3** | +2.1 |
| **Overall** | **Avg. Pass@8** | **64.9** | **77.0** | **+12.2** |

> 🌟 在科学类任务上提升尤为显著，说明 ESRL 能更广泛覆盖成功的复杂推理路径。

---

### 强指令模型上的持续增益
在更强的 **Qwen3-30B-A3B-Instruct** 上继续训练：

| 方法 | Avg. Pass@1 | Δ | Avg. Pass@8 | Δ |
|------|------------|----|------------|----|
| GRPO | 63.4 | — | 77.3 | — |
| **ESRL** | **68.9** | **+5.5** | **79.0** | **+1.7** |

> 即使在已高度调优的 instruct model 上，ESRL 仍带来显著提升，证明其探索价值独立于初始能力。

---

### 消融实验结果

#### （1）自适应噪声 vs 固定噪声
| 方法 | Avg. Pass@1 | Avg. Pass@8 |
|------|------------|-------------|
| Non-adaptive (fixed σ) | 38.7 | 59.6 |
| **Adaptive (entropy-based)** | **42.1** | **64.2** |

✅ 自适应机制至关重要，尤其在 AIME 和 AMC 上差异明显。

#### （2）锚定专家数量与候选池大小（Kanchor, Mexplore）
| 配置 | Avg. Pass@1 | Avg. Pass@8 |
|------|------------|-------------|
| Kanchor=4, Mexplore=16 | **42.1** | 64.2 |
| Kanchor=2, Mexplore=16 | 41.6 | **64.9** |
| Mexplore=64（过大） | ↓ 性能下降 |

> 发现：
- 更多锚定专家 → 更高 Pass@1（保精度）
- 更多探索槽位 → 更高 Pass@8（扩覆盖）
- 候选池不宜过大，否则引入劣质路径

#### （3）噪声注入位置
| 策略 | Avg. Pass@1 |
|------|------------|
| Only last 2 layers | 41.7 |
| First 8 layers | 41.5 |
| **All layers** | **42.1** |

✅ 全层扰动效果最好，但后期层扰动影响更大（见 Fig. 9）。

#### （4）解耦训练分析
| 策略 | Avg. Pass@1 | Avg. Pass@8 |
|------|------------|-------------|
| Joint (T>0 + noise) | 42.1 | **64.2** |
| Decoupled (先 noise 后 T) | **42.4** | 63.0 |

> 有趣发现：**仅靠 routing noise 就能完成有效 RL 训练**，且 decoupled 方式在高预算下表现更好，说明 expert-path 探索本身具备完整学习能力。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Expert routing 是有效的 RL 探索维度**  
   扰动 router logits 可显著改变 next-token 分布，类似于提高 temperature，是 token-level sampling 的互补途径。

2. ✅ **受控扰动优于盲目扰动**  
   无约束扰动会激活不合适专家，损害质量；而 **anchored sampling + entropy-adaptive noise** 能在多样性与质量间取得良好平衡。

3. ✅ **ESRL 显著提升 rollout 效率**  
   - 在更小 rollout 组（64 samples）下即可超越 baseline 在大组（128/256）的表现；
   - 维持更高的 **informative group ratio**，提供更强的学习信号。

4. ✅ **改善专家利用均衡性**  
   图 10 显示 ESRL 有效缓解了 RL 训练中常见的 “expert collapse” 问题，维持更低的 load imbalance。

5. ✅ **完全正交且可组合**  
   ESRL 作用于 rollout 阶段，不修改 reward 或 loss，未来可与其他 RL 改进方法（如 GSPO、RO-GRPO）结合。

---

### 局限性
1. **超参数敏感性**：尽管有自适应机制，但 `σ_max`、`Kanchor`、`Mexplore` 等仍需调优。
2. **仅适用于 MoE 模型**：无法应用于 dense 模型，应用范围受限。
3. **未探索反向信用分配**：当前方法未尝试优化 router 本身，仅用于探索。

---

### 未来工作方向
1. **联合优化 router 与 policy**：将 routing 决策也纳入 RL 学习过程，实现端到端的 expert-path learning。
2. **动态调整探索策略**：根据任务难度或中间状态动态决定是否探索、如何探索。
3. **扩展至 vision-language 或 multimodal MoE**：验证在非文本模态下的有效性。
4. **与 curriculum learning 结合**：早期鼓励更多探索，后期收敛到最优路径。

---

> 💡 **总体评价**：该论文首次系统论证了 **expert routing 作为空间探索维度的有效性**，提出的 ESRL 框架设计精巧、实证充分，在多个 MoE 架构和任务上实现了稳定且显著的性能提升，为 MoE-based LLM 的 RL 训练开辟了新的研究方向。

</details>

---

### 4. [Unleashing the Power of Equality Saturation for Tensor Program Superoptimization](https://arxiv.org/abs/2609.12330)

**Authors**: Qi Zhan, Xing Hu, Xin Xia, Shanping Li  
**Category**: cs.DC  
**Published**: 2026-09-14  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.12330v1  

#### Abstract
Efficient GPU implementations of tensor programs often require joint optimization of high-level algebraic formulations and low-level execution strategies. However, the resulting search space grows rapidly as transformations combine across operators, making joint optimization difficult to scale. We p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Unleashing the Power of Equality Saturation for Tensor Program Superoptimization*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代深度学习模型（尤其是大语言模型）对 GPU 上的 **tensor program** 执行效率要求极高。然而，高效的实现通常需要同时优化：
- **高层代数表达式**（如数学等价变换）
- **底层执行策略**（如 tiling、fusion、并行化）

传统方法在联合优化这两者时面临挑战：
- 搜索空间随算子数量指数级增长，难以扩展；
- 多数工具要么只关注代数重写（如 TASO），要么只处理调度（如 TVM），缺乏统一框架；
- 现有基于 equality saturation 的方法（如 TENSAT、Trinity）在表示能力和搜索效率之间难以平衡。

### 🚀 提出的新方法：**EQUIFORGE**
EQUIFORGE 是一个基于 **equality saturation** 的 tensor program superoptimizer，其核心创新包括：

#### （1）统一的 IR 设计（Unified IR）
- 引入一种纯函数式表达语言，**同时表示高层 tensor 表达式和低层 tiled 计算**。
- 支持 `MatMul`, `Reduce`, `Partition`, `Combination`, `Collection` 等操作，并通过类型系统跟踪张量形状和平行状态（parallel state）。
- 允许在同一个 e-graph 中进行代数重写与并行细化（parallel refinement），实现跨层次的组合优化。

#### （2）可组合的 equality rules
设计了一套模块化的规则体系，支持：
- **Algebraic equalities**：如结合律、分配律（`MatMul(X, W) * a = MatMul(X*a, W)`）
- **Parallel refinement**：将 `MatMul` 映射为 `TileMatMul + Combination/Reduce`
- **Reduction fusion**：融合独立或依赖的 reduction（使用 Neptune 的 repair 函数处理依赖情况）
- **Propagation rules**：将分区传播到上下游操作中，形成端到端的 tiled 实现
- **Intermediate storage**：通过 `Collection` 决定是否将中间结果物化到全局内存

这些规则可以**自动组合**，直接从原始表达式推导出类似 FlashAttention 的 fused kernel。

#### （3）早期压缩（Early Compaction）
- 在 extraction 阶段前识别并剪枝功能等价但结构不同的候选程序（如加法交换顺序不同）。
- 定义等价关系 ~，在 partial program 层面进行剪枝，避免重复构建完整候选。
- 显著降低 extraction 成本，提升搜索效率。

#### （4）子图组合优化（Subgraph Composition）
- 将大图划分为多个连通子图，分别进行 equality saturation 和 extraction。
- 各子图生成的候选可复用，最后通过接口组合成完整程序。
- 平衡了优化覆盖范围与搜索开销。

---

### 🔍 相比现有方法的优势

| 方法 | 局限性 | EQUIFORGE 的改进 |
|------|--------|------------------|
| **TVM / TensorIR** | 依赖手动调度模板，难以探索代数变换 | 自动发现代数+执行联合优化 |
| **TASO / PET** | 仅限于图级替换，不支持低层实现生成 | 统一高层表达与底层实现搜索 |
| **Trinity** | 基于显式循环的 tile-level IR，代数重写受限 | 使用纯表达式 IR，代数规则更易组合 |
| **TENSAT** | 未考虑低层 GPU 实现细节 | 融合 tiling、fusion、memory 策略 |
| **Mirage / Prism** | 枚举式搜索，成本高 | 基于 e-graph 紧凑表示，支持大规模搜索 |

> ✅ 总结：EQUIFORGE 实现了 **“从数学公式到高效 kernel” 的端到端自动优化**，是首个将 equality saturation 应用于 full-stack tensor program superoptimization 的系统。

---

## 2. 核心实验方法和设置

### 📊 数据集与工作负载
涵盖多种典型 tensor program 和 Transformer 层结构：

#### （1）通用 Tensor Programs
- `nGPT`: 小型前馈网络
- `RMSNorm-MLP`, `RMSNorm-SwiGLU`, `GatedMLP`
- `LoRA`（低秩适配）
- `LayerNorm+GEMM`

#### （2）注意力机制（MHA）
- **Decode 模式**（自回归生成）：B ∈ {1, 8, 16}
- **Prefill 模式**（上下文编码）：B ∈ {1, 8, 16}

#### （3）复杂 Transformer 层（Case Studies）
- QK-normalized MLA
- mHC
- Sliding-window GQA
- Attention sinks
- Differential attention

所有配置均使用 Llama 2 或 BERT 的典型参数规模。

---

### ⚙️ 实验设置

| 项目 | 设置 |
|------|------|
| **硬件平台** | NVIDIA A100 80GB PCIe / RTX 5090 |
| **测量方式** | 使用 CUDA Graphs 测量中位延迟，输出误差容忍 L2 < 0.01 |
| **搜索预算** | 每个配置最多 4 小时（含 saturation + extraction + tuning） |
| **前端输入** | PyTorch 2.10 程序，通过 `torch.export` 提取计算图 |
| **后端代码生成** | Triton 3.6 |
| **equality saturation 引擎** | EGG 0.11 |

---

### 📈 评估指标
- **性能加速比**：相对于最强 baseline 的几何平均加速（geometric mean speedup）
- **最大加速比**
- **搜索效率**：e-graph 规模、extraction 时间、候选数量
- **正确性验证**：与 PyTorch 输出对比

---

### 🆚 基线方法对比

#### 非注意力任务：
- **PyTorch eager**
- **torch.compile**
- **TVM 0.26 (MetaSchedule)**：1000 trials
- **Mirage**：同搜索时间限制
- **Trinity**：默认配置

#### 注意力任务：
- **FlashAttention**
- **FlashInfer**
- **Neptune**
- **Trinity**

> 注：Prism 因未开源未参与比较。

---

## 3. 主要实验结果和性能指标

### 📈 总体性能表现（Figure 7 & 8）

| 指标 | 结果 |
|------|------|
| **几何平均加速比** | **1.32×** 超过最强 baseline |
| **最高加速比** | **2.74×**（RMSNorm-MLP @ B=8） |
| **Decode 模式下 vs FlashAttention** | 最高 **1.87× 更快** |
| **Prefill 模式下 vs FlashAttention** | 接近性能，差距仅 **3–6%** |

> 💡 说明 EQUIFORGE 不仅能复现 FlashAttention 级别的优化，还能进一步超越。

---

### 🔬 典型案例加速效果

| 模型 | 加速比（vs torch.compile） | 关键优化技术 |
|------|----------------------------|-------------|
| **QK-normalized MLA** | **3.16×** | 移动 key projection 到 query 路径，避免历史张量展开 |
| **mHC** | **5.84×** | 跨组件融合 residual update 与 normalization |
| **Sliding-window GQA** | **1.06×** | 单 kernel 实现窗口内 online softmax |
| **Attention sinks** | **1.48×** | 融合 sink token 与近期 token 的 attention 计算 |
| **Differential attention** | **1.51×** | 混合权重计算与归一化融合 |

> ✅ 这些案例表明 EQUIFORGE 能发现人类专家级别的复杂优化模式。

---

### 🔍 搜索效率分析（Table 2 & Figure 10）

#### e-graph 规模（B=16）
| 工作负载 | E-classes / E-nodes |
|---------|--------------------|
| RMSNorm-MLP | 346 / 949 |
| MHA decode | 5,223 / 22,339 |
| MHA prefill | 5,286 / 65,283 |

> 表明 attention 类任务搜索空间极大。

#### Extraction 剪枝效果（Figure 10）
三种策略对比：
1. **No compaction**：枚举所有候选
2. **After completion**：先生成再去重
3. **Early compaction**：在 partial program 阶段剪枝

| 工作负载 | 枚举候选数 → 去重后 | Early compaction 加速倍数 |
|----------|---------------------|----------------------------|
| nGPT | 69.96M → 126 | **148×** |
| RMSNorm-MLP | 53.53M → 72 | **128×** |
| RMSNorm-SwiGLU | 69.06k → 29 | **>1000×**（另一策略超时） |
| LayerNorm+GEMM | 62.86M → 478 | **2,465×** |

> ✅ **Early compaction 最多减少 2,465 倍的 expansion 数量**，显著提升搜索可行性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Equality saturation 可有效统一高层代数优化与低层执行优化**  
   —— 通过统一 IR 和可组合规则，实现了从数学表达式到高性能 kernel 的自动推导。

2. **EQUIFORGE 能自动发现 FlashAttention 级别的优化结构**  
   —— 包括 online softmax、incremental normalization、fused reduction 等。

3. **早期压缩极大提升了搜索效率**  
   —— 在不影响最终质量的前提下，大幅削减冗余候选的构造成本。

4. **子图划分使大规模程序优化成为可能**  
   —— 支持跨 layer boundary 的融合优化，适用于复杂 Transformer 架构。

5. **在多样 workload 上实现稳定且显著的性能提升**  
   —— 几何平均 **1.32×**，最高 **5.84×**，优于主流编译器和 superoptimizer。

---

### ⚠️ 局限性
1. **搜索时间仍较长**：尽管有 early compaction，prefill 场景仍需近 30 分钟 saturation。
2. **依赖 Triton 后端**：最终性能受 Triton codegen 调度能力影响（如 FlashAttention 差距部分源于此）。
3. **修复函数合成有限**：目前仅支持特定形式的 dependent reduction fusion。
4. **未支持分布式优化**：当前聚焦单设备 kernel 优化。

---

### 🔮 未来工作方向
1. **引入 sketch-guided rewriting**：利用用户提示引导搜索方向，加快收敛。
2. **集成 learned cost model**：预测候选性能，优先探索高潜力路径。
3. **扩展至 multi-GPU / distributed setting**：结合 Unity 等工作，支持并行策略联合优化。
4. **支持更多硬件后端**：如 CUDA C++、Metal、ROCm。
5. **自动化 rule synthesis**：从性能日志中反向挖掘新的 equality rules。

---

## ✅ 总结
EQUIFORGE 成功展示了 **equality saturation 在 tensor program superoptimization 中的强大潜力**。它不仅能够：
- 自动生成媲美甚至超越人工设计的高性能 kernel（如 FlashAttention-style），
- 还能在统一框架下融合代数变换、tiling、fusion、memory 策略，
- 并通过 early compaction 和 subgraph composition 实现可扩展搜索。

该工作为下一代 AI 编译器提供了一个全新的范式：**以语义等价推理为核心驱动程序结构演化**，有望成为连接算法创新与工程极致性能的关键桥梁。

</details>

---

### 5. [Correlation-Guided Fast Machine Unlearning via Hessian Analysis](https://arxiv.org/abs/2609.12620)

**Authors**: Ayushi Thakur, Ruchir Gupta, Amit Kumar Jaiswal, Prayag Tiwari  
**Category**: cs.LG  
**Published**: 2026-09-14  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.12620v1  

#### Abstract
The increasing adoption of machine learning in network and distributed security systems has created an urgent need for mechanisms that can selectively and efficiently remove the influence of specific training data to eliminate compromised or adversarial data points from production models. Privacy re...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Correlation-Guided Fast Machine Unlearning via Hessian Analysis**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文针对 **machine unlearning**（机器遗忘）中的一个核心瓶颈：**计算效率低下**。  
在现实场景中（如网络安全系统、GDPR合规），需要快速移除特定训练数据的影响，但现有的基于 **influence function** 的近似遗忘方法依赖昂贵的 **Hessian-inverse-vector product** 计算，每删除一个数据点都需要重复此操作，导致难以扩展。

尤其当多个相关数据点需被连续遗忘时（如入侵检测中的一组恶意样本），这种重复计算成为严重瓶颈。

---

### 🚀 提出的新方法与核心思想

作者提出了一种 **基于相似性的高效遗忘框架**，其核心创新如下：

#### （1）**利用 Pearson Correlation 识别高相关数据点**
- 引入 **Pearson correlation coefficient** $ \rho $ 作为数据点之间的相似性度量。
- 发现：对于高度相关的数据点 $ x $ 和 $ z $，它们对模型参数的影响方向（即梯度）是成比例的：  
  $$
  \nabla f(x, w^*) \approx \alpha \nabla f(z, w^*)
  $$

#### （2）**推导闭式参数更新规则（Closed-form update rule）**
- 利用 **Sherman-Morrison formula** 和 **Hessian damping** 技术，从首次遗忘的结果中直接推导后续相似点的遗忘更新：
  $$
  w_x = w_z + \frac{\alpha + 1}{1 - s_\lambda} (w_z - w^*)
  $$
  其中：
  - $ w_z $：已遗忘点 $ z $ 后的模型参数
  - $ s_\lambda $：damped self-influence score
  - $ \alpha $：由 Pearson correlation 导出的比例因子

> ⚡ 这避免了为每个新点重新计算 Hessian-inverse-vector product。

#### （3）**理论保障与数值稳定性设计**
- 引入 **Hessian damping**（$ H \leftarrow H + \lambda I $）确保矩阵可逆且条件数可控。
- 推导了 **parameter update error bound**，证明误差随输入维度多项式增长（而非指数），适用于高维场景。
- 给出安全阈值 $ \rho_{\min} $，用于控制最大允许近似误差。

---

### 🔍 相比现有方法的优势

| 方面 | 优势 |
|------|------|
| **计算效率** | 对后续相关遗忘请求实现 **82× wall-clock speedup**，将复杂度从多次 $ O(d^3) $ 降为一次 $ O(d^3) $ + 多次 $ O(d) $ 操作 |
| **模型效用保留** | 在遗忘后保持更高 accuracy（相比 SOTA 提升达 $10^{-2}$） |
| **隐私保护能力** | 忘记效果优于基线，MIA 攻击成功率更低，ToW 分数更高 |
| **适用性广** | 不依赖强凸性假设，在非线性深度模型（CNN, ResNet）上依然有效 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

共评估 **7个多样化数据集**，涵盖分类与回归任务：

| 类型 | 数据集 | 模型架构 |
|------|--------|----------|
| 回归 | California Housing (8维特征) | Linear Regression |
| 回归 | Diabetes (scikit-learn) | Linear Regression |
| 分类 | Synthetic GMM (合成数据) | 3-layer FCN |
| 分类 | MNIST | CNN / Logistic |
| 分类 | Fashion-MNIST | CNN |
| 分类 | CIFAR-10 | ResNet-18 |
| 分类 | LFW (人脸) | ResNet-18 |
| 分类 | CIFAR-100 | ResNet-50（大规模验证） |

所有特征均标准化（zero mean, unit variance）。

---

### 🧪 实验设置

- **训练配置**：
  - 分类任务使用 Adam，学习率 0.001~0.07，epoch 数 30–100
  - 回归任务使用 SGD + L2 正则化（$ \lambda = 0.01 $）
  - Damping 参数 $ \lambda $ 设为与正则化相同值

- **遗忘策略**：
  - 模拟 **batch-sequential unlearning** 场景（连续删除多个相关样本）
  - 批大小 $ k \in \{5,10,20,50\} $

---

### 📊 评估指标

| 指标 | 描述 |
|------|------|
| **Avg. Acc. (AR)** | 遗忘后在保留数据集上的平均准确率（越高越好） |
| **Acc. Unlearn (AU)** | 忘记质量度量：$ 1 - \|w - w_{\text{std}}\|^2 / \max(\cdot) $，越接近1表示越接近标准遗忘结果 |
| **Tug-of-War (ToW)** | 衡量遗忘平衡性：综合 forget/retain/test 集表现差异（越高越好） |
| **Membership Inference Attack (MIA)** | 攻击者判断某样本是否属于训练集的成功率（越低越好，表示忘记更彻底） |
| **Wall-clock time** | 单次遗忘耗时（衡量效率） |
| **Condition number $ \kappa(H) $** | 评估 Hessian 数值稳定性 |

---

### 🆚 基线方法对比

| 基线 | 简介 |
|------|------|
| **Retrain from scratch** | 完全重训，金标准但极慢 |
| **MITR** | 基于信息论正则化的遗忘方法 |
| **RUM(A)/RUM(B)** | 当前 SOTA 的遗忘方法，强调可扩展性 |
| **Hessian-free Unlearning** | 无需 Hessian 的认证遗忘方法 |

> 注：部分方法不支持 CNN/ResNet 架构，故某些实验缺失结果。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### （1）**效率提升显著**

| 方法 | 平均单点时间 | 相比加速倍数 |
|------|-------------|--------------|
| Retrain from scratch | 29.02 s | — |
| Standard influence unlearning | 169.54 ms | — |
| **Proposed (Ours)** | **2.07 ms** | **≈82× faster** |

> ✅ 仅需一次 Hessian-inverse-vector 计算，后续通过闭式公式完成。

---

#### （2）**遗忘效果优越**

在 **CIFAR-10 + ResNet-18** 上与 SOTA 对比：

| Method | ToW ↑ | MIA ↓ |
|--------|-------|------|
| RUM(A) | 0.715 | 0.489 |
| RUM(B) | 0.920 | 0.590 |
| **Ours** | **0.950** | **0.660** |

> 💡 虽然 MIA 略高，但 ToW 更优，说明在“保留有用知识”与“彻底遗忘”之间取得更好平衡。

---

#### （3）**模型效用更强**

在多个数据集上，**accuracy retention (AR)** 显著高于基线：

- 在 **CIFAR-100 + ResNet-50** 上：
  - 我们的方法：**AR = 91.7%**, **AU = 88.4%**（k=20）
  - RUM(B)：AR = 84.52%, AU = 76.86%
  - ➕ **+7.18% 效用增益**

---

#### （4）**相似性度量有效性验证**

在 Diabetes 数据集上比较三种相似性度量：

| Similarity Measure | Win Rate (最小误差次数占比) |
|--------------------|----------------------------|
| Cosine Similarity | 23.4% |
| Projection-based | 33.3% |
| **Pearson Correlation** | **43.2%** ✅ |

> ✅ Pearson 表现最佳，因其对中心化线性关系建模更契合梯度空间特性。

---

#### （5）**消融实验与稳定性分析**

##### 条件数改善（CIFAR-10）

| $ \lambda $ | $ \kappa(H^*) $ | $ \kappa(H_\lambda) $ | 改善倍数 |
|------------|------------------|------------------------|---------|
| $10^{-2}$ | $1.88\times10^{13}$ | 49,735 | $3.78\times10^8\times$ |

> ✅ 加入 damping 后，Hessian 条件数从不稳定降至安全范围（< $10^8$），保证数值稳定。

##### 参数敏感性（Table X）
- 最佳 $ \lambda = 10^{-2} $：AR=84.1%, ToW=0.950
- $ \lambda $ 过大 → 过度正则化 → 性能下降

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Pearson correlation 是预测梯度相似性的最优指标**，优于 cosine 和 projection-based 方法（43.2% 赢率）。
2. **相似数据点的遗忘可以共享影响方向**，使得后续遗忘可通过缩放首次更新来高效逼近。
3. **Hessian damping 是实现数值稳定的必要手段**，能将病态 Hessian 转换为良态矩阵。
4. **所提方法在真实场景下兼具高效性与有效性**：
   - 速度提升 **82×**
   - 模型效用提升 **>7%**
   - 忘记质量（ToW）达到 **0.950**
5. **理论误差界具有实际指导意义**，虽保守约 35–39 倍，但仍可用于设定安全阈值 $ \rho_{\min} $。

---

### ⚠️ 局限性

1. **依赖数据点间的线性相关性假设**：若数据分布高度非线性或稀疏，则相似性估计可能失效。
2. **适用于“批内相关”的连续遗忘场景**，对完全独立的数据点仍需原始 influence 计算。
3. **未考虑对抗性攻击下的 unlearning 安全性**（如伪造高相关样本干扰遗忘过程）。
4. **当前分析基于二次损失近似**，在极端非凸情况下可能存在偏差。

---

### 🔮 未来工作方向

1. **自适应切换机制**：动态判断何时使用 similarity-based 近似，何时回退到完整 unlearning。
2. **扩展至联邦学习环境**：在分布式节点间实现高效的协同遗忘。
3. **增强对抗鲁棒性**：防御针对相似性计算的投毒攻击。
4. **结合 exact unlearning 方法**（如 SISA）构建混合遗忘系统。
5. **探索其他高效矩阵更新技巧**（Beyond Sherman-Morrison）以处理更大规模更新。

---

## 总结

> 本文提出了一种 **基于 Pearson correlation 和 Hessian 分析的快速 machine unlearning 框架**，通过识别相关数据点并复用其影响方向，实现了 **82× 的推理加速**，同时在 **accuracy retention** 和 **forgetting effectiveness** 上全面超越 SOTA 方法。该方法具备坚实的理论基础、良好的数值稳定性，并已在多种模型和数据集上得到验证，为 GDPR 合规、网络安全系统中的实时数据删除提供了实用解决方案。

</details>

---

### 6. [Efficient AI Model Deployment Using Quantization Analysis Tool](https://arxiv.org/abs/2609.11954)

**Authors**: Dwith Chenna, Kanishka Macherla  
**Category**: cs.LG  
**Published**: 2026-09-14  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.11954v1  

#### Abstract
As deep learning models are increasingly deployed on resource constrained devices, the demand for efficient model optimization techniques continues to grow. Effective deployment of AI models on edge and low power platforms requires optimization methods that reduce model size and computational cost w...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Efficient AI Model Deployment Using Quantization Analysis Tool*

## 1. 论文的主要贡献和创新点

### 解决了什么问题
随着深度学习模型在边缘设备（edge devices）和资源受限平台（如嵌入式系统、移动设备、NPU/GPU）上的广泛应用，模型部署面临**计算能力、内存带宽和功耗预算有限**的挑战。虽然量化（Quantization）是降低模型大小和推理延迟的有效手段，但其引入的**量化误差可能导致显著的精度下降**，尤其是在敏感层中。

现有方法通常采用统一的量化策略（uniform quantization），缺乏对各层量化敏感性的细粒度分析，导致在精度与效率之间难以做出最优权衡。此外，开发者缺乏系统化工具来理解量化影响、识别问题操作并指导混合精度（mixed-precision）配置。

### 提出了什么新方法或新思路
本文提出了一种名为 **Quantization Analysis Tool** 的系统性框架，基于 ONNX 构建，用于支持高效的 AI 模型部署。该工具的核心创新包括：

- **Layer-wise Sensitivity Analysis（逐层敏感性分析）**：通过逐层单独量化并测量其对整体精度的影响，识别出对低精度最敏感的关键层。
- **Weight & Activation 分布可视化**：提供直方图形式的权重和激活值分布对比（FP32 vs 量化后），帮助理解动态范围压缩、剪裁（clipping）和偏态分布等问题。
- **统一的 ONNX 基础平台**：利用 ONNX 的跨框架兼容性，实现从 PyTorch/TensorFlow 等训练框架导出模型后的标准化分析流程。
- **可操作的量化洞察（Actionable Insights）**：结合统计分析与可视化，为开发者提供明确建议，例如跳过某些高敏感层的量化或采用选择性量化（selective quantization）。

### 相比现有方法的优势
| 方面 | 现有方法局限 | 本工具优势 |
|------|---------------|------------|
| 分析粒度 | 多为全模型或粗粒度分析 | 支持 operation-level 和 layer-wise 细粒度分析 |
| 可解释性 | 缺乏直观可视化支持 | 提供分布图、敏感性排序等可视化报告 |
| 决策支持 | 依赖经验调参 | 数据驱动地推荐混合精度策略 |
| 跨平台兼容性 | 通常绑定特定框架 | 基于 ONNX，支持多框架模型输入 |
| 部署前验证 | 很少提供预部署分析 | 允许在部署前“profile”量化行为 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **Calibration Dataset**：ImageNet 的子集，用于量化校准（calibration），捕捉激活值的真实分布。
- **Validation Dataset**：标准 ImageNet 测试集的一部分（如 100 张图像用于快速评估趋势），用于评估量化前后模型的 Top-1 准确率变化。

### 实验设置和评估指标
#### 模型架构
选取多个主流 CNN 架构进行测试，涵盖轻量级与通用模型：
- ResNet-18 / ResNet-50
- MobileNet-V2 / MobileNet-V3-Large
- EfficientNet-B0 / B1
- SqueezeNet 1.0 / 1.1
- ShuffleNet V2 (0.5x, 1.0x)

#### 量化配置
- 采用 Post-Training Quantization (PTQ) 范式
- 数据类型：INT8（默认）
- 支持 per-tensor 和 per-channel 量化模式
- 工具允许用户自定义 calibration method（如 Min-Max, Moving Average）

#### 评估指标
| 指标 | 描述 |
|------|------|
| **Top-1 Accuracy** | 主要性能指标，衡量分类任务准确率 |
| **Accuracy Drop** | 量化后相对于 FP32 基线的精度损失 |
| **Model Size Reduction** | 量化带来的存储节省（隐含在精度-大小权衡曲线中） |
| **Layer-wise Sensitivity Score** | 每层单独量化后的输出偏差或精度下降程度 |
| **Weight/Activation Distribution Skew** | 分析分布形态对量化鲁棒性的影响 |

#### 基线方法对比
- **Baseline (FP32)**：原始浮点模型精度
- **Standard PTQ**：直接应用默认 ONNX Runtime 的 PTQ 流程
- **Tool-Guided PTQ**：使用本工具分析后，手动排除高敏感层或调整部分层精度的量化方案

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table I）

| Model | Baseline Accuracy | Standard PTQ | Tool-Guided PTQ | 提升幅度 |
|-------|-------------------|-------------|------------------|---------|
| ResNet-18 | 69.5% | 69.0% | 69.0% | +0.0% |
| ResNet-50 | 80.0% | 77.2% | 77.4% | +0.2% |
| MobileNet-V2 | 70.7% | 67.3% | 68.7% | +1.4% |
| MobileNet-V3-Large | 74.1% | 69.2% | 71.3% | +2.1% |
| **EfficientNet-B0** | **77.1%** | **37.4%** | **54.9%** | **+17.5%** ✅ |
| EfficientNet-B1 | 77.4% | 73.5% | 73.9% | +0.4% |
| SqueezeNet 1.0 | 58.6% | 57.6% | 58.3% | +0.7% |
| **ShuffleNet V2 0.5** | **60.4%** | **43.4%** | **57.0%** | **+13.6%** ✅ |
| ShuffleNet V2 1.0 | 68.6% | 61.9% | 65.6% | +3.7% |

> 🔍 **关键观察**：
> - 对于原本 PTQ 效果较差的模型（如 EfficientNet-B0、ShuffleNet V2 0.5），**工具引导下的量化带来了巨大提升（最高达 17.5%）**
> - 这些模型在标准 PTQ 下出现严重退化（<40%），说明其结构对量化高度敏感
> - 工具成功识别出关键敏感层（仅占 ~5% 的操作），通过保留这些层为高精度即可大幅恢复性能

### 与基线方法的对比结果
- 在多数模型上，**Tool-Guided PTQ 显著优于 Standard PTQ**，尤其在非典型或高度优化的网络结构中效果更明显。
- ResNet 类模型本身对量化较鲁棒，因此增益较小；而 MobileNet、EfficientNet、ShuffleNet 等轻量模型因大量使用 depthwise conv 和非线性结构，更容易受量化扰动。
- 工具不仅提升了精度，还揭示了“**并非所有层都需要高精度**”，从而支持更灵活的 mixed-precision 设计。

### 消融实验结果（隐含分析）
尽管未明确列出消融表，文中通过以下方式进行了机制验证：
- **Layer-wise 敏感性扫描**（Fig 6）显示：
  - ResNet：早期和晚期层相对稳定，中间层可安全量化
  - MobileNet V2：初始和最后几层量化时出现明显 accuracy dip
  - ShuffleNet 1.1：中间层敏感，后期反而稳定 → 表明不同架构具有独特敏感模式
- **分布分析发现**：
  - 权重误差普遍低于激活误差 → 强调 calibration dataset 和 method 的重要性
  - 高敏感层常表现出 **skewed 或 heavy-tailed 分布**，易发生 clipping
  - 对称 vs 非对称量化对某些层影响显著

---

## 4. 关键结论和发现

### 论文的主要发现
1. **并非所有层对量化同等敏感**：少数关键层（<10 层，占比约 5%）主导了量化误差，尤其是：
   - EfficientNet 中的最后 `Gemm` 层和 `Mul` 操作
   - SqueezeNet/ShuffleNet 的早期卷积层（编码基础特征）
2. **统一量化策略次优**：uniform INT8 量化在复杂或紧凑模型中会导致灾难性精度下降，必须结合 layer-wise 分析。
3. **激活分布比权重更重要**：activation 的动态范围和 outlier 更容易导致量化失真，强调高质量 calibration 的必要性。
4. **可视化 + 敏感性分析 = 可解释性增强**：工具提供的 histogram 和 sensitivity ranking 极大提升了开发者对模型行为的理解。
5. **选择性量化（Selective Quantization）有效且可行**：只需保护少量高敏感层，即可在保持高效的同时显著恢复精度。

### 方法的局限性
- 当前主要支持 CNN 架构，尚未验证在 Transformer、LLM 上的效果（作者已在 Future Work 中提及）。
- 敏感性分析过程需要运行多次 partial quantization 推理，带来一定计算开销（非完全自动化搜索）。
- 依赖 ONNX 导出质量，某些复杂算子可能无法正确解析。
- 尚未集成硬件性能建模（如 latency estimation on specific NPU），决策仍偏重精度维度。

### 未来工作方向
1. 扩展支持 **Transformer 和 Large Language Models (LLMs)** 的量化分析
2. 引入 **自动化的 mixed-precision search 算法**，基于 sensitivity score 自动分配 bit-width
3. 添加更多 metric comparators（如 latency、energy consumption）
4. 结合 **hardware-specific backend models**（如 ONNX Runtime with EPs）进行真实设备性能预测
5. 支持更先进的量化技术：如 **asymmetric clipping、learned step size、block-wise quantization**

---

📌 **总结一句话**：  
本文提出的 **Quantization Analysis Tool** 通过构建一个基于 ONNX 的、集成了 layer-wise sensitivity analysis 与 distribution visualization 的分析平台，显著提升了量化模型的部署效率与精度表现，特别是在传统 PTQ 失效的轻量模型上实现了高达 **+17.5% 的精度回升**，为资源受限场景下的 AI 部署提供了强有力的数据驱动决策支持。

🔗 **项目开源地址**：[https://github.com/dwithchenna/onnx-analyzer](https://github.com/dwithchenna/onnx-analyzer)

</details>

---

### 7. [Fixed State, Long Reach: What a Constant-Size Cache Buys Block Diffusion at Scale](https://arxiv.org/abs/2609.11998)

**Authors**: Vaibhav Singh, Pierre-Andr\'e No\"el, Torsten Scholak, Eugene Belilovsky, Oleksiy Ostapenko  
**Category**: cs.LG  
**Published**: 2026-09-14  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.11998v1  

#### Abstract
Diffusion language models decode tokens in parallel, but their bidirectional denoiser rules out the naive key--value (KV) cache behind fast autoregressive inference. Block diffusion restores caching by decoding block-by-block, and the block caches deployed on it so far are tied to attention: O(L)in ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Fixed State, Long Reach: What a Constant-Size Cache Buys Block Diffusion at Scale**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
传统 **Diffusion Language Models (DLMs)** 虽然能并行解码多个 token，但由于其**双向注意力机制**，无法像自回归模型（AR）那样使用高效的 **Key-Value (KV) Cache**，导致每次去噪步骤都需要重新计算整个序列的注意力，推理延迟高、内存开销大。

尽管 **Block Diffusion** 通过块级自回归生成恢复了部分缓存能力，但已有实现仍依赖于 **Attention-based KV Cache**，其内存占用和计算复杂度随上下文长度 $O(L)$ 增长，在长上下文场景下依然昂贵。此外，许多方法是“训练即插即用”（training-free retrofit），其缓存仅为真实双向计算的近似，影响精度。

### **提出的新方法与新思路**
本文提出了一种**统一的、精确的块缓存机制**，基于三种不同 backbone 构建可缓存的 block-diffusion 模型，并首次在大规模（3B 参数）上系统比较：

- **Attn**: 全注意力 backbone
- **Mamba**: 纯双向 Mamba-2 backbone（基于 SSM）
- **Hybrid**: 注意力与 Mamba 混合结构（每五层插入一个 Attention 层）

核心创新在于：
- 利用 **State-Space Models (SSMs)** 如 Mamba 的**固定大小隐藏状态**（fixed-size recurrent state）作为 block cache，实现 $O(1)$ 内存与延迟。
- 所有模型均采用 **block-causal 训练目标**（single-frontier objective），确保训练与推理时的缓存行为一致，使缓存为**精确计算而非近似**。
- 统一接口支持三类 backbone 的缓存机制，便于公平比较。

### **相比现有方法的优势**
| 方面 | 优势 |
|------|------|
| **缓存效率** | Mamba 的 cache 是 $O(1)$，而 Attention 是 $O(L)$，长上下文下内存和延迟显著更低 |
| **扩展性** | 固定内存 footprint 支持更大 batch 和更长上下文，Attention 在 batch>1 时迅速 OOM |
| **检索能力** | Mamba/Hybrid 可外推至训练长度 8–16×，Attention 在 2× 即崩溃 |
| **质量一致性** | 缓存机制与训练目标对齐，避免 approximation error |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **预训练数据**：`Nemotron-CC` 数据集，共 **300B tokens**
- **评估任务**：
  - **NIAH (Needle-in-a-Haystack)**：测试长距离信息检索能力
  - **LongBench**：16 项英文与代码相关的长上下文理解任务
  - **下游任务**：8 个常识推理任务（如 PIQA、ARC-c、BoolQ 等）
  - **生成质量**：Generative Perplexity（Gen-PPL）、MAUVE 分数

### **实验设置**
- **模型规模**：所有模型均为 **3B 参数级**，共享以下配置：
  - `d_model = 2560`, `depth = 28`, `block size G = 32`
  - 训练序列长度：1024 tokens
  - 优化器：AdamW（bf16），global batch size 4096，总训练步数 ~71.5k，覆盖约 300B tokens
- **训练目标**：Single-frontier block-diffusion objective —— 每次仅对一个“前沿块”进行去噪，其余前缀干净、后缀全掩码
- **硬件平台**：单张 **NVIDIA H100 80GB GPU**，使用 cudagraph 消除 kernel 启动开销

### **评估指标**
| 类别 | 指标 |
|------|------|
| **效率** | per-step latency（ms）、peak memory（GB）、decode throughput（tok/s） |
| **长上下文性能** | NIAH 准确率、LongBench 宏平均得分 |
| **生成质量** | Gen-PPL、MAUVE |
| **下游性能** | 多项选择任务准确率（via block-diffusion likelihood harness） |

### **基线方法对比**
- **主对比模型**：
  - **Attn**（类似 BD3LM）
  - **Mamba**
  - **Hybrid**
- **附加变体**：
  - Attn + NTK-RoPE：训练即插即用的位置编码缩放技术，用于延长可用上下文
- 所有模型在相同训练目标、数据、超参下训练，唯一变量是 **sequence mixer**（即 backbone 类型）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **效率表现（最长达 256k tokens）**

| 模型 | 上下文长度 | Latency (ms/step) | Memory (GB) | Throughput (tok/s) |
|------|------------|-------------------|-------------|--------------------|
| **Attn** | 256k | 29.31 | 81.91 | 114 |
| **Hybrid** | 256k | 10.74 | 21.52 | 231 |
| **Mamba** | 256k | **6.76** | **7.65** | **297** |

> 🔥 **Mamba 相比 Attn 提升**：
> - **延迟降低 4.3×**
> - **内存减少 11×**
> - **吞吐提升 2.6×（单流）**

#### ✅ **批处理扩展性（Batch Scaling）**
- Mamba 的 per-stream 内存恒定，因此可轻松扩展 batch size：
  - 在 batch=8、L=256k 时仍稳定运行，aggregate throughput 达 **1593 tok/s**
- Attn 在 batch=2 时已 OOM，最大 aggregate throughput 仅 114 tok/s（batch=1）
- ➜ **Mamba 实现 14× 更高的聚合吞吐量**

#### ✅ **长上下文检索能力（NIAH）**

| Context Length ($\times$ train len) | Attn | Hybrid | Mamba |
|-------------------------------------|------|--------|-------|
| 1k ($1\times$) | 100% | 100% | 99.0% |
| 2k ($2\times$) | 12.3% | 53.4% | **75.7%** |
| 8k ($8\times$) | 0% | 23.0% | **23.6%** |
| 16k ($16\times$) | 0% | 2.3% | **22.3%** |

> 💡 Mamba 在 **16× 训练长度** 下仍保持超过 20% 的检索准确率，而 Attention 完全失效。

#### ✅ **现实任务表现（LongBench）**

| Context Length | Attn | Hybrid | Mamba |
|----------------|------|--------|-------|
| 2k | 7.07 | 10.83 | **11.20** |
| 4k | 4.65 | 10.39 | **10.45** |
| 8k | 3.96 | 9.30 | **10.46** |
| 16k | 3.64 | 9.13 | **10.18** |

> 📈 Mamba 几乎不随长度下降，而 Attn 性能减半以上；Hybrid 表现稳健，接近 Mamba。

#### ✅ **下游与生成质量**

| 指标 | Attn | Hybrid | Mamba |
|------|------|--------|-------|
| 下游任务 Macro-Accuracy | 0.434 | 0.432 | 0.421 |
| Gen-PPL @ S=16 | 9.22 | 9.06 | 9.50 |
| MAUVE @ S=16 | 0.236 | 0.289 | 0.287 |

> ✅ **质量无显著损失**：Hybrid 与 Attn 质量持平，Mamba 仅略低 ~1 point，但换来完全常量内存推理。

#### ✅ **消融分析：参数 vs 架构**

| 模型 | 参数量 | FLOPs/token @ L=1024 | FLOPs/token @ L=64k |
|------|--------|------------------------|----------------------|
| Attn | 3.033B | 5.712G | 24.209G |
| Hybrid | 3.360B (+11%) | 6.139G | 9.442G |
| Mamba | 3.431B (+13%) | 6.232G | **6.232G** |

> 🔍 尽管 Mamba 多出 13% 参数，但其 FLOPs 不随 $L$ 增长，而 Attention 的 $O(Ld)$ 项主导长上下文成本。证明收益来自 **architecture 而非 parameter count**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **SSM-based cache 实现真正 $O(1)$ 推理**  
   Mamba 的 recurrent state 天然适合作为 block cache，带来**恒定内存与延迟**，突破 Attention 的 $O(L)$ 瓶颈。

2. **长上下文泛化能力源于线性状态机制**  
   Mamba/Hybrid 能外推到 **8–16× 训练长度**，而 Attention 在 2× 即崩溃，说明位置无关的 recurrence 更适合长度外推。

3. **效率提升无质量代价**  
   Hybrid 模型在效率与质量之间取得完美平衡：保留部分 Attention 以维持全局 recall，同时大幅降低内存与延迟，且下游性能与 Attn 持平。

4. **架构优势远超参数差异**  
   即便 Mamba 多出 13% 参数，其长上下文下的 FLOPs 仍远低于 Attn，证明性能增益来自 **mixer 架构本身**。

### **局限性**
- 所有模型均在 **1024 长度**上训练，长上下文评估属于**外推**（extrapolation），非原生长上下文训练。
- 未结合 confidence-aware parallel decoding 或 adaptive caching 等进一步加速策略。
- 当前 focus 在 block-level caching，尚未探索 intra-block 的细粒度优化。

### **未来工作方向**
- 结合 RoPE-scaling（如 NTK-RoPE、YaRN）或直接在长上下文上训练，提升绝对长程性能。
- 将 exact cache 与 Fast-dLLM 类似的 confidence-aware 并行解码结合，进一步提升速度。
- 探索更多 hybrid 结构设计（如动态切换 attention/Mamba）。
- 将该范式推广至更大规模模型（>10B）和多模态扩散模型。

---

> **一句话总结**：  
> 本文证明，基于 **Mamba 的 state-space cache** 可为 block diffusion 带来 **常量内存、恒定延迟、强外推能力**，在 256k 上实现 **14× 吞吐优势**，且不牺牲生成质量，为高效长上下文 DLMs 提供了坚实基础。

</details>

---

### 8. [Chopthin-Consensus Power Sampling: A Diversity-Preserving Approach to LLM Decoding](https://arxiv.org/abs/2609.12243)

**Authors**: Minoo Ahmadi, Seyedarmin Azizi, Erfan Baghaei Potraghloo, Mehdi Kamal, Massoud Pedram  
**Category**: cs.CL  
**Published**: 2026-09-14  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.12243v1  

#### Abstract
Inference-time power sampling via Sequential Monte Carlo (SMC) can substantially improve large language model (LLM) reasoning without requiring post-training. However, many existing SMC approaches rely on equal-weight resampling, which can aggressively prune low-weight trajectories, discarding poten...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Chopthin-Consensus Power Sampling: A Diversity-Preserving Approach to LLM Decoding

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

在基于 **Sequential Monte Carlo (SMC)** 的大语言模型（LLM）推理过程中，标准的 **equal-weight resampling**（如系统性重采样）虽然能缓解权重退化（weight degeneracy），但会严重损害**基因多样性（genealogical diversity）**。这种做法通过强制等权并复制高权重粒子、丢弃低权重粒子，可能导致那些当前权重较低但最终正确的推理路径被永久删除。

此外，传统的 **weight-based selection**（如按权重抽样）倾向于选择高权重路径，而忽视了由多样性保留下来的低权重正确路径，造成“生成”与“选择”的不匹配。

---

### 提出的新方法与新思路

作者提出 **Chopthin-Consensus Power Sampling (CCPS)**，包含两个核心创新：

#### （1）**Chopthin Resampler**：多样性保留型重采样
- 替代传统 equal-weight resampling，采用 **Chopthin** 算法。
- 不强制所有输出粒子权重相等，而是**限制最大与最小权重之间的比率**（bounded weight ratio）。
- 对粒子分类处理：
  - **Thin**：极轻粒子以概率存活，避免完全随机删除；
  - **Keep**：中等权重粒子原样保留；
  - **Chop**：过重粒子被切分为多个等权子粒子。
- 优势：
  - 保持 SMC 近似的无偏性（unbiasedness）；
  - 保证有效样本量（ESS）下限；
  - 显著减少基因多样性损失（lineage collapse）。

#### （2）**Semantic-Majority Selection**：语义多数投票机制
- 在最终答案选择阶段，不再依赖权重抽样（weight draw），而是：
  1. **Merge**：合并 token 完全相同的轨迹，防止重复计票；
  2. **Cluster**：将语义等价的答案聚类（无需参考黄金标签）；
     - 数学题使用表达式归一化与符号简化；
     - 编程任务使用程序在自生成测试输入上的行为一致性进行聚类。
  3. **Vote**：返回支持轨迹数量最多的聚类代表答案。
- 实现了对多样性的充分利用，提升最终准确率。

---

### 相比现有方法的优势

| 维度 | 传统方法（Power-SMC） | CCPS |
|------|------------------------|-------|
| **Resampling** | Equal-weight, 强制重置权重 | Chopthin, 保留不等权重，控制比例上限 |
| **Diversity Preservation** | 严重丢失低权重路径 | 更多保留潜在正确路径 |
| **Selection Strategy** | Weight draw, 忽视低权重正确路径 | Semantic majority, 利用路径数量而非权重 |
| **理论性质** | 扰动整个群体 | 仅干预极端粒子，保持中间稳定 |
| **适用性** | 通用但有偏差风险 | 可扩展至代码生成等复杂场景 |

> ✅ **核心优势**：CCPS 在不增加训练成本的前提下，通过改进 inference-time 的 resampling 和 selection 机制，显著提升了 LLM 推理能力。

---

## 2. 核心实验方法和设置

### 使用的数据集

共五个 benchmark，涵盖数学推理与代码生成：

| 数据集 | 类型 | 题目数 | 特点 |
|--------|------|--------|------|
| **MATH500** | 数学应用题 | 500 | 中学至竞赛级数学问题 |
| **GSM8K** | 小学数学应用题 | 1,319 | 多步逻辑推理 |
| **AIME 2022–2024** | 竞赛数学 | 90 | 高难度证明与构造题 |
| **GPQA Diamond** | 博士级科学问答 | 198 | 四选一，需深度领域知识 |
| **HumanEval** | Python 编程 | 164 | 函数补全任务 |

---

### 实验设置

- **模型**：
  - `Qwen2.5-Math-7B`
  - `Qwen2.5-7B`
  - `Qwen3-4B`

- **解码参数**：
  - 粒子数 $ N = 32 $
  - Sharpening exponent $ \alpha = 2 $
  - Proposal temperature $ t = 0.5 $
  - ESS 触发阈值 $ K = 0.5 $
  - Block size $ B = 64 $ tokens
  - α-ramp：前 100 tokens 从 1 线性增长到 2

- **Chopthin 参数**：
  - Ratio bound $ n = 3 + \sqrt{8} \approx 5.83 $，确保 ESS 下限为 ~0.5N

- **对比基线**：
  - **Power-SMC (Systematic + Weight Draw)**：原始 SMC 方法
  - **Baseline decoding**：普通温度采样（$ T=1 $）
  - **Low-temperature decoding**：低温采样（$ T=1/\alpha $）
  - **MH Power Sampling**：Metropolis-Hastings 方法（文献值）

---

### 评估指标

| 指标 | 定义 | 意义 |
|------|------|------|
| **Final Answer Accuracy** | 正确答案占比 | 实际可用性能 |
| **Oracle Coverage** | 至少一个粒子包含正确答案的比例 | “天花板”性能，衡量多样性保留程度 |
| **Distinct Trajectories (D)** | 去重后不同路径数量 | 衡量基因多样性 |
| **Surviving Root Lineages (Rt)** | 存活祖先路径数 | 衡量谱系多样性 |
| **ESS (Effective Sample Size)** | 权重分布均匀性度量 | 衡量重采样有效性 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2）

| Model | Method | MATH500 | GSM8K | AIME | GPQA | HumanEval |
|-------|--------|---------|-------|------|------|-----------|
| Qwen2.5-Math-7B | Power-SMC | 77.0 | 89.5 | 15.6 | 30.3 | 58.5 |
| | **CCPS (Ours)** | **81.4** | **90.8** | **16.7** | **33.8** | **61.6** |
| Qwen2.5-7B | Power-SMC | 74.2 | 91.0 | 10.4 | 29.3 | 73.2 |
| | **CCPS (Ours)** | **76.0** | **91.4** | **12.2** | **30.3** | **76.8** |
| Qwen3-4B | Power-SMC | 79.0 | 90.1 | 15.6 | 29.8 | 71.3 |
| | **CCPS (Ours)** | **81.8** | **92.1** | 15.6 | **40.4** | 70.7 |

> 🔺 **最高提升达 +10.6 个百分点**（Qwen3-4B on GPQA）

---

### 与基线方法的对比结果

- 在 **15 个实验设置中**：
  - **Oracle Coverage 提升**：**13/15** 设置中高于 Power-SMC
  - **Final Accuracy 提升**：**14/15** 设置中等于或优于 Power-SMC
- 即使在 Power-SMC 表现不佳的情况下，CCPS 仍能通过多样性找回正确答案。

#### 典型案例分析（Figure 3a）：
- 在 `Qwen2.5-Math-7B` 上，CCPS 的 oracle coverage 达 **87.0%** vs. Power-SMC 的 **84.6%**
- 最终 accuracy 提升 **+4.4pp**

---

### 消融实验结果（Ablation Studies）

#### （1）Ratio Bound $ n $ 敏感性分析（Figure 3a）
- 方法对 $ n $ 鲁棒，在广泛范围内均优于 baseline
- 默认值 $ n = 3+\sqrt{8} $ 同时达到最佳 accuracy 与 coverage
- 更宽松的 bound 有助于保留更多 distinct trajectories

#### （2）Carried Weights vs. Uniform Reset（Figure 3b）
| 方法 | Oracle Coverage | Majority Accuracy |
|------|------------------|--------------------|
| Systematic Resampling | 84.6 | 79.4 |
| Hybrid (Chopthin 分配 + 重置权重) | 85.8 | 81.2 |
| **CCPS (完整版)** | **87.0** | **81.4** |

> ✅ 结论：性能增益来自两部分：
> - **约一半** 来自 offspring allocation（即 Chopthin 的生存策略）
> - **另一半** 来自 carried weights（携带不等权重向前传播）

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **Equal-weight resampling 是推理性能瓶颈之一**  
   它虽恢复 ESS，却牺牲了 genealogical diversity，导致潜在正确路径丢失。

2. ✅ **Chopthin 能有效保留多样性**  
   通过 bounded-weight resampling，显著提高 oracle coverage（+1.2 ~ +4.4pp），说明更多正确路径得以存活。

3. ✅ **Semantic-Majority Selection 是释放多样性的关键**  
   若继续使用 weight draw，则 Chopthin 的优势无法体现；只有结合语义投票，才能将“存在正确路径”转化为“返回正确答案”。

4. ✅ **多样性保留 + 多样性感知选择 是互补机制**  
   二者协同作用，实现无需训练即可媲美甚至超越 RL post-training 的效果。

5. ✅ **CCPS 在多个开放模型上稳定有效**  
   跨越 3 个模型、5 个 benchmark，表现出强泛化能力。

---

### 方法的局限性

1. **计算开销略高**  
   Chopthin 的求解阈值过程比系统性重采样稍慢（约 0.2ms/event），但在整体延迟中占比极小（< 5×10⁻⁴），可忽略。

2. **依赖高质量聚类机制**  
   Semantic clustering 的效果受限于 grader 或执行环境的设计，尤其在模糊语义或部分正确答案场景下可能失效。

3. **未解决根本不确定性建模问题**  
   仍基于 importance sampling 框架，未引入显式的 uncertainty modeling 或 verifier guidance。

4. **Oracle Coverage 仍有 Gap**  
   如 Qwen3-4B on MATH500 中，baseline decoding 准确率为 82.0%，但 CCPS oracle coverage 为 84.8%，说明仍有 2.8% 的潜力未被 selector 捕获。

---

### 未来工作方向

1. **结合 verifier 或 reward model**  
   在 selection 阶段引入轻量级 verifier，进一步提升聚类与投票质量。

2. **动态调整 ratio bound $ n $**  
   根据任务难度或中间状态自适应调节重采样强度。

3. **扩展至其他 inference-time scaling 方法**  
   如 marginal sharpening、Twisted SMC 等，探索 Chopthin 的通用性。

4. **应用于多模态或规划任务**  
   将 CCPS 框架推广至视觉推理、机器人决策等长序列生成场景。

5. **研究更优的 consensus aggregation 方式**  
   例如加权投票、置信度融合、graph-based consensus 等。

---

> 📌 **总结一句话**：  
> **CCPS 通过 Chopthin 保留更多可能正确的推理路径，并通过 semantic-majority 投票将其转化为实际性能提升，实现了无需训练的高效 LLM 推理增强。**

</details>

---

### 9. [SAS: Simple Attention Sparsification via End-to-End Optimization of Context Ranking](https://arxiv.org/abs/2609.13141)

**Authors**: Zhiwei Li, Lei Zhu, Hao Gu, Xiang Hu, Yan Wang, Haitao Mi, Sirui Han, Leo Liang, Zhijiang Guo  
**Category**: cs.CL  
**Published**: 2026-09-14  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.13141v1  

#### Abstract
Post-training attention sparsification reduces the quadratic cumulative attention cost of pretrained Transformers by selecting a small set of context units (tokens or blocks) for each query. Existing trainable methods usually use a lightweight selector to score context units, followed by hard Top-K ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# SAS: Simple Attention Sparsification via End-to-End Optimization of Context Ranking — 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

在大型语言模型（LLMs）中，**长上下文推理**的效率瓶颈在于自回归生成过程中注意力机制的**二次计算成本**（cumulative attention cost）。为缓解此问题，**post-training attention sparsification** 被提出，即在预训练后通过稀疏化注意力来减少计算量。

然而，现有可训练的稀疏方法通常依赖于轻量级选择器（selector）对上下文单元打分，并通过 **hard Top-K 选择**确定关注的上下文块。由于 Top-K 是非可微操作，梯度无法从语言建模损失（language modeling loss）反向传播到选择器，导致这些方法普遍采用**逐层蒸馏原始模型的稠密注意力分布**作为监督信号。

这种做法存在**ranking misalignment**（排序错位）问题：选择器被训练去模仿原始模型“在哪里注意”，而非学习“哪些上下文真正影响最终预测”。这可能导致有限的注意力预算被浪费在对预测无益的上下文上。

---

### ✅ 提出的新方法与新思路

论文提出了 **Simple Attention Sparsification (SAS)**，一种通过端到端优化上下文排序（context ranking）实现简单高效的注意力稀疏化方法。

#### 核心思想：
将选择器的连续得分（continuous scores）注入到注意力计算中，使其成为可微路径的一部分，从而允许语言建模损失直接通过标准反向传播优化选择器。

#### 关键设计选择（使该简单设计有效）：
1. **Log-space Gate Injection**  
   将选择器输出的分数以 `log g` 形式加到注意力 logits 中（`softmax(qK + log g)`），而非作用于 softmax 输出。这使得门控直接影响注意力权重分配，而非仅缩放输出值。

2. **Normalized Gate Activation**  
   使用 `softmax` 对历史块的得分进行归一化，确保其相对重要性是经过校准的，避免因未归一化导致的数值不稳定或饱和。

3. **Preservation of Continuous Scores**  
   在训练时保留软门控（soft gating），不将其坍缩为硬 Top-K 掩码。这样模型不仅能学到“选哪些”，还能学到“优先级如何”。

4. **Efficient Sparse Training Scope**  
   仅对选中的块进行注意力计算，大幅降低训练开销，同时实验证明其最终性能与全范围训练相当。

5. **Memory-Efficient Triton Kernel**  
   实现了一个融合门控的 FlashAttention 风格 Triton 内核，在 tile 级别完成 `qK^T + log g` 的计算，避免显式构建完整注意力矩阵，支持长序列训练。

---

### ✅ 相比现有方法的优势

| 维度 | 传统方法（如 SeerAttention-R） | SAS |
|------|-------------------------------|-----|
| 训练目标 | 层级注意力蒸馏（distillation） | 直接语言建模损失优化 |
| 梯度流 | 被 Top-K 阻断，需辅助任务 | 可微，端到端优化 |
| 上下文排序质量 | 对齐原始注意力分布 | 对齐最终预测效果 |
| 是否需要教师模型 | 是 | 否 |
| 实现复杂度 | 高（需双模型训练） | 低（单目标优化） |

> ✅ **优势总结**：SAS 去除了对教师注意力或辅助蒸馏目标的依赖，实现了更直接、更有效的上下文选择策略学习。

---

## 2. 核心实验方法和设置

### 📚 数据集

- **训练数据**：OpenR1-Math-220k（约 93.7K 数学推理样本）
- **评估任务**：
  - **Reasoning**：MATH500, GPQA-Diamond, AIME24, AIME25
  - **Long-context Understanding**：LongBench（多语言、多任务）
  - **Agentic Tasks**：BFCL (Multi-Turn), VitaBench（真实场景工具调用）
  - **扩展测试**：RULER（超长上下文能力）

### ⚙️ 实验设置

- **模型架构**：基于 Qwen3 系列（4B, 8B, 14B），冻结主干网络，仅训练 AttnGate 类型的选择器。
- **稀疏设置**：
  - Block size: 64
  - Top-K: 32（即最多关注 2048 个 token）
  - 注意力预算（budget）：1024, 2048, 4096
- **训练细节**：
  - 序列长度：32,768
  - 优化器：AdamW，lr=1e-3，cosine decay
  - 分布式训练框架：VeOmni + FSDP
- **推理实现**：集成至 SGLang，利用 paged KV cache 和 FlashInfer 支持高效解码。

### 🔁 基线方法对比

| 方法 | 类型 | 特点 |
|------|------|------|
| **Full Attention** | 全注意力 | 原始稠密模型，性能上限 |
| **Sliding Window / StreamingLLM** | 无训练、固定模式 | 局部窗口 + sink token |
| **Quest** | 查询感知、无训练 | 动态稀疏，启发式规则 |
| **SeerAttention-R** | 可训练、蒸馏驱动 | 当前最强 post-training baseline，使用相同 selector 架构 |

> 所有方法在相同 backbone、selector 结构、训练数据下比较，公平性强。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（代表性结果）

#### ✅ Reasoning 任务（Qwen3-4B, budget=2048）

| 方法 | MATH500 ↑ | GPQA-Diamond ↑ | AIME24 ↑ |
|------|-----------|----------------|----------|
| Full Attn | 93.93 | 56.19 | 71.25 |
| SeerAttention-R | 91.85 | 49.94 | 55.83 |
| **SAS** | **93.47** | **54.86** | **68.85** |
| **提升** | +1.62 | **+4.92** | **+13.02** |

> 在最难的 AIME24 上，SAS 比蒸馏方法高出 **13 分以上**，接近甚至超越全注意力表现。

#### ✅ LongBench（Qwen3-14B, budget=4096）

| 输入长度 | SeerAttn-R | SAS | Full Attn |
|--------|------------|-----|---------|
| <8K | 55.7 | 56.2 | 56.6 |
| >8K | 53.5 | 54.8 | 55.0 |
| **平均** | 55.7 | 56.2 | 56.6 |

> SAS 在长输入段（>8K）提升显著（+1.3），几乎恢复全注意力性能。

#### ✅ Agentic 任务（BFCL Multi-Turn, Qwen3-4B, budget=2048）

| 方法 | Score ↑ |
|------|--------|
| Full Attn | 35.75 |
| SeerAttn-R | 29.00 |
| **SAS** | **32.50** |
| **提升** | **+3.5** |

> 在多轮交互任务中，SAS 显著优于基线，且在 budget=4096 时几乎追平全注意力。

---

### 🔍 消融实验结果（Ablation Study）

在 GPQA-Diamond 上进行控制变量分析（Qwen3-4B, budget=2048）：

| 设计选择 | 最终准确率（%） | 关键发现 |
|--------|----------------|----------|
| Outer Gate | 41.6 | 不如 inner gate，说明门控应参与 softmax 归一化 |
| Inner Gate + Sigmoid | 17.0 | 未归一化导致饱和，性能崩溃 |
| Inner Gate + Raw Logits | 18.8 | 无 softmax 归一化，难以区分重要性 |
| **Inner + Softmax + Soft Gate** | **54.47** | 完整 SAS 设计，性能最优 |
| Sparse Scope Training | 54.8 | 初期慢但最终性能持平，训练更高效 |

> ✅ 实验验证了四个设计选择的必要性，尤其是 **log-space + softmax normalization + soft gating** 的组合至关重要。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **端到端优化显著优于蒸馏**  
   直接使用语言建模损失优化选择器，能学到更符合下游任务需求的上下文排序，尤其在低预算下优势巨大。

2. **跨层互补性更强**  
   SAS 学到的选择模式在不同层之间更具互补性，其跨层并集与“全注意力 Oracle”重叠度更高（见 Figure 5b），而蒸馏方法倾向于每层都复制局部注意力分布。

3. **生成更短、更少截断**  
   在 reasoning 任务中，SAS 生成的答案更简洁，截断率更低（Figure 6），表明其能更快定位关键信息完成推理。

4. **解码速度大幅提升**  
   在 512K 上下文下，SAS 达到 **5.6× 解码加速**（batch=1），batch=8 时可达 **~13×**，且延迟基本不随上下文增长。

5. **可迁移至 continued pretraining**  
   在 OLMo3-7B 上联合训练 backbone 和 selector，SAS 仍优于 HiLS-Attn 和滑窗基线，证明其通用性。

---

### ⚠️ 局限性

1. **极端长上下文性能下降**  
   在 RULER 128K 测试中，尽管 SAS 优于 SeerAttention-R，但仍远低于全注意力（如 Qwen3-4B 从 63.81↓到 21.87 @128K），表明当前 block summary 机制可能丢失细粒度信息（如 needle-in-a-haystack）。

2. **选择阶段成瓶颈**  
   随着上下文增长，Top-K 排序和 selector scoring 成为主要耗时环节（占 step 时间 90% @512K），未来需优化选择内核。

3. **依赖 block-level abstraction**  
   方法基于 block 粒度压缩，可能不适合需要 token 级精确访问的任务。

---

### 🔮 未来工作方向

1. **设计更强大的 block summarization 机制**  
   如引入可学习的 block embedding 或 attention-over-blocks，以保留更多语义细节。

2. **动态调整 Top-K 数量**  
   根据 query 复杂度自适应分配注意力预算。

3. **进一步优化 selection kernel**  
   开发更高效的 Top-K 和 selector scoring 内核，突破当前性能瓶颈。

4. **探索 SAS 在 pretraining 阶段的应用**  
   从头训练稀疏注意力模型，而非 post-training 适配。

---

## 总结

✅ **SAS 是一个简洁而强大的 post-training attention sparsification 新范式**：

- 它通过 **端到端优化语言建模损失** 替代传统的注意力蒸馏；
- 利用 **log-space soft gating** 实现梯度流动；
- 并结合 **归一化、连续评分保留、稀疏训练范围** 等关键设计，实现高效且高性能的稀疏注意力；
- 在 reasoning、long-context、agentic 等多种任务上全面超越现有方法，尤其在低预算下优势显著；
- 同时具备良好的工程实用性，已通过 Triton kernel 实现高效部署。

> 🔥 **一句话总结**：SAS 表明，最有效的上下文选择不是模仿“模型怎么注意”，而是直接学习“什么能让模型答对”。

</details>

---

### 10. [Clustering-Based Balanced Sampling and Allocation with Data Parallelism for High-Performance Fine-Tuning](https://arxiv.org/abs/2609.12584)

**Authors**: Hyunjin Kim, Youngeun Nam, Jaemin Han, Wonhyeok Choi, Jae-Gil Lee  
**Category**: cs.LG  
**Published**: 2026-09-14  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.12584v1  

#### Abstract
Instruction-tuning datasets for large language models (LLMs) are often large, redundant, and imbalanced, limiting efficient adaptation. Naive large-batch fine-tuning repeatedly includes overrepresented sample groups while weakly covering underrepresented but informative ones, especially under data p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Clustering-Based Balanced Sampling and Allocation with Data Parallelism for High-Performance Fine-Tuning

## 1. 论文的主要贡献和创新点

### 解决了什么问题
大型语言模型（LLMs）在进行 **instruction tuning** 时，常面临训练数据冗余、分布不均衡的问题。传统的随机采样（random sampling）在 **Data Parallelism (DP)** 设置下会导致：
- 高频样本组被重复处理，造成计算冗余；
- 低频但信息量大的样本覆盖不足；
- 各 GPU worker 间梯度信号相似，降低优化多样性。

这限制了训练效率和稳定性，尤其是在大规模分布式训练中。

---

### 提出了什么新方法或新思路
作者提出了 **CluSTER**（Cluster-aware balanced Sampling framework for Training Efficient data Reduction），一个面向 **DP instruction tuning** 的聚类感知平衡采样框架，其核心思想包括：

1. **Gradient-Space Clustering**  
   在梯度空间对样本进行聚类，将具有相似更新方向的样本归为一类，从而捕捉语义和学习动态上的共性。

2. **Dual-Level Coverage**  
   - **Inter-level coverage**：不同 GPU worker 分配来自不同 cluster 的样本，提升跨 worker 的梯度多样性；
   - **Intra-level coverage**：每个 cluster 内选择远离质心的“边界”样本（peripheral selection），增强簇内多样性。

3. **Balanced Sampling + Weighted Update**  
   - 对所有 cluster 下采样至最小簇大小，实现负载均衡；
   - 引入 **cluster-weighted gradient update**，使每个 cluster 的梯度贡献与其原始规模成正比，保留原始数据分布。

4. **轻量级梯度代理构建**  
   使用 final-layer hidden states 和 token-level cross-entropy 构建梯度代理嵌入（gradient-proxy embedding），避免昂贵的全参数反向传播。

---

### 相比现有方法的优势
| 方法 | 局限性 | CluSTER 的优势 |
|------|--------|----------------|
| Random Sampling | 易受长尾分布影响，梯度方差大 | 显著降低冗余，提升梯度多样性 |
| Uniform Sampling | 要求预定义平衡簇，难以应用于文本数据 | 自动从梯度空间发现语义簇 |
| IFD / LESS / S2L | 需要大量预处理（如 influence score、小模型训练轨迹） | 仅需一次前向传播，复杂度更低 |
| 其他数据选择方法 | 忽略 DP 中的 worker 分配机制 | 显式考虑 DP-aware 分配，优化多 GPU 协同 |

> ✅ **核心优势**：在显著减少训练时间的同时，几乎无损模型性能，并且预处理开销极低。

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 类型 | 规模 | 领域 |
|-------|------|------|------|
| **Magicoder-OSS-Instruct-75K** | Code instruction tuning | 75,197 样本 | 编程 |
| **Evol-Instruct-Code-80K** | Code instruction tuning | 78,264 样本 | 编程 |
| **MedInstruct-52K** | Medical instruction tuning | 52,002 样本 | 医疗问答 |

---

### 实验设置
- **模型**：
  - CodeLlama-Python-7B / 13B
  - Llama-2-7B / 13B
- **硬件**：4–8 × H100 GPU，采用 PyTorch DDP 或 FSDP 进行 **Data Parallelism**
- **训练配置**：
  - Global batch size: 512
  - 学习率：5e-5（Adafactor）或 2e-5（AdamW）
  - Epochs: 2–3
- **CluSTER 参数**：
  - 聚类数 $ K $ = GPU 数量
  - 聚类容量上限 $ \alpha = 1.5 $
  - 选择比例 $ r \in \{1.00, 0.75, 0.50\} $

---

### 评估指标
| 任务 | 指标 |
|------|------|
| Code Generation | **Pass@1** on HumanEval(+), MBPP(+) |
| Medical QA | **Accuracy** on MedMCQA, MedQA, PubMedQA, MMLU-medical subsets |
| 效率指标 | 训练时间（hours）、数据选择时间、总耗时 |

---

### 基线方法对比
| 类别 | 方法 | 描述 |
|------|------|------|
| 采样类 | Random Sampling | 默认策略 |
|       | Uniform Sampling | 理想化均匀采样 |
| 选择类 | IFD | 基于指令遵循难度过滤 |
|       | LESS | 基于梯度影响估计选择 |
|       | S2L | 利用小模型训练轨迹选择数据 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 📊 表格摘要（以 Magicoder-OSS-Instruct-75K + CodeLlama-7B 为例）

| 方法 | 数据使用率 | Avg. Pass@1 | 训练时间 (hr) |
|------|------------|-------------|----------------|
| Full Random | 100% | 57.6 | 1.13 |
| CluSTER (r=1.00) | 79% | **56.8** | **0.92** |
| CluSTER (r=0.75) | 59% | **56.5** | **0.69** |
| CluSTER (r=0.50) | 40% | **55.9** | **0.47** |

> 🔥 **最高提速达 69.6%**（从 1.13h → 0.34h），性能损失 < 2%

---

#### 医疗领域表现（MedInstruct-52K + Llama-2-7B）

| 方法 | 数据使用率 | 平均准确率 | 训练时间 |
|------|------------|-----------|----------|
| Full Random | 100% | 48.5 | 0.99h |
| CluSTER (r=0.50) | 31% | **48.0** | **0.30h** |

> ⏱️ 使用仅 **31% 数据**，达到接近全量训练的性能，**训练时间减少 69.6%**

---

### 与基线方法的对比结果

| 方法 | 性能 | 训练效率 | 预处理成本 |
|------|------|----------|------------|
| Random | 较低 | 一般 | 最低 |
| Uniform | 中等 | 中等 | 高（需先验知识） |
| IFD / LESS / S2L | 中高 | 一般 | 极高（需 backward 或 proxy training） |
| **CluSTER** | ✅ **最高** | ✅ **最优** | ✅ **最低**（仅一次 forward） |

> 在多个数据集和模型尺度上，**CluSTER 在所有数据预算下均优于所有基线方法**。

---

### 消融实验结果

#### （1）加权机制的作用（Weighting Mechanism）
| 方法 | HumanEval+ | MBPP+ | Avg. |
|------|------------|--------|------|
| w/o weighting | 52.3 | 66.8 | 55.5 |
| **w/ weighting** | **54.9** | **67.0** | **56.8** |

> 加权机制带来 **+1.3 pts 提升**，验证其对保持原始分布的重要性。

#### （2）簇内选择策略比较
| 策略 | Gradient Diversity | Avg. Score |
|------|--------------------|------------|
| Core-centric | 低 | 55.3 |
| Random | 中 | 56.1 |
| **Peripheral** | ✅ **最高** | ✅ **56.8** |

> “边缘样本”选择显著提升梯度多样性和最终性能。

#### （3）超参数敏感性分析
- **选择比例 r**：即使降至 25%，性能下降平缓；
- **聚类数 K**：对 GPU 数量变化鲁棒；
- **聚类容量 α**：在合理范围内稳定。

> 表明 CluSTER 不依赖精细调参，易于部署。

---

## 4. 关键结论和发现

### 主要发现
1. **梯度空间聚类能有效揭示语义结构**  
   - 发现的 clusters 与任务类别（algorithm, SQL, math, syntax）高度一致（82% 匹配）；
   - 支持“相似梯度 ≈ 相似语义意图”的假设。

2. **Dual-level coverage 是高效训练的关键**  
   - Inter-level：避免多 worker 处理相同 cluster；
   - Intra-level：选择边界样本提升簇内多样性；
   - 二者结合可最大化每步的有效信息增益。

3. **CluSTER 实现“少而精”的训练范式**  
   - 减少最多 **69.6% 训练时间**；
   - 几乎无损模型性能（Pass@1 / Accuracy 下降 < 2%）；
   - 同时大幅降低数据选择开销（相比 LESS/S2L 快 10× 以上）。

4. **方法具备良好扩展性**
   - 支持从 4 到 8 GPU 的横向扩展；
   - 在 7B 和 13B 模型上均有效；
   - 跨编程与医疗领域通用性强。

---

### 方法的局限性
1. **单节点限制**  
   当前实验局限于单节点多 GPU，未测试跨节点、更大 batch size 场景下的通信开销影响。

2. **数据集规模有限**  
   实验集中在数十万样本级别，尚未验证在百万级以上超大规模 instruction 数据中的表现。

3. **聚类质量依赖梯度代理**  
   使用 final-layer 梯度代理可能无法完全反映深层非线性动态，极端情况下可能导致误聚类。

4. **静态聚类假设**  
   聚类在训练前一次性完成，未考虑训练过程中样本重要性的动态变化。

---

### 未来工作方向
1. **Dynamic Clustering**  
   探索在训练过程中周期性更新聚类，适应模型状态演化。

2. **Multi-Node Extension**  
   结合 ZeRO 或 FSDP 进一步优化跨节点通信与负载均衡。

3. **Integration with Curriculum Learning**  
   将 cluster 权重与课程学习结合，逐步引入难样本。

4. **Application to Other Modalities**  
   扩展至 vision-language 或 audio 模型的 instruction tuning。

---

> 💡 **总结一句话**：  
> **CluSTER 通过梯度空间聚类 + 双层平衡采样 + 加权更新，在不牺牲性能的前提下，实现了高达 70% 的训练加速，是当前最高效的 instruction tuning 数据选择方案之一。**

GitHub 开源地址：[https://github.com/kaist-dmlab/CluSTER](https://github.com/kaist-dmlab/CluSTER)

</details>

---

### 11. [Performance, Efficiency and Collapse -- Advantages and Challenges in Offline Post-training of Code LLMs](https://arxiv.org/abs/2609.11956)

**Authors**: Abhinav Anand, Sanjana Reddy Pachika, Shweta Verma, Mira Mezini  
**Category**: cs.LG  
**Published**: 2026-09-14  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.11956v1  

#### Abstract
Post-training with reinforcement learning (RL) is a critical phase in the development of code-generating large language models (LLMs), as it ensures adherence to instructions and the production of functionally correct code. This process typically requires computationally intensive code sample genera...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Performance, Efficiency and Collapse -- Advantages and Challenges in Offline Post-training of Code LLMs**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决了什么问题
传统的 **Reinforcement Learning with Verifiable Rewards (RLVR)** 在 Code LLM 的 post-training 阶段存在以下瓶颈：
- **计算开销大**：需要在线从模型中采样大量代码并执行验证（GPU-CPU 通信频繁）。
- **效率低下**：Transformer 架构推理慢，导致训练周期长。
- **反馈信号单一**：通常只优化功能正确性（functional correctness），难以引入效率、安全性等多维反馈。

该论文提出：**能否完全离线地进行 RL post-training？即不依赖实时采样，而是直接利用已有的带奖励标签的代码数据集进行训练。**

### ✅ 提出的新方法/新思路
- **首次系统研究了 Offline RL 在 Code LLM post-training 中的可行性与稳定性**。
- 采用 **Offline Reinforcement Learning (Offline RL)** 框架，基于预收集的数据集（如 CodeNet）进行策略优化，无需在训练过程中生成新样本。
- 使用 **RLOO + GRPO-style advantage normalization** 的组合目标函数，在保持简单性的同时实现稳定训练。
- 引入对 **logit variance** 和 **logit gap** 的分析，揭示了 Offline RL 不稳定的根本原因，并提出早期停止（early stopping）策略来缓解模型崩溃（model collapse）。

### ✅ 相比现有方法的优势
| 维度 | 传统 Online RL | 本文 Offline RL |
|------|----------------|------------------|
| **采样方式** | 在线采样 + 执行验证 | 完全离线，使用已有数据 |
| **计算成本** | 高（需反复生成 & 执行代码） | 极低（单卡 GPU 即可完成） |
| **训练速度** | 慢（小时级甚至天级） | 快（几小时以内） |
| **可扩展性** | 受限于采样吞吐 | 易于扩展至多语言、多反馈维度 |
| **环保性** | 能耗高 | 更节能 |

> 💡 **核心优势**：实现了高效、低成本、零在线采样的 post-training，为未来支持多种反馈信号（如效率、安全）提供了可能路径。

---

## 2. **核心实验方法和设置**

### 📚 数据集
- **训练数据**：`CodeNet` 数据集
  - 包含约 8,321 个 Python 编程任务提交记录。
  - 每条包含：问题描述、Python 代码、执行状态（通过/失败/超时等）。
  - 奖励映射如下：
    ```
    +1.0 → All Test Cases Passed  
    -0.1 → Test Cases Failed  
    -0.5 → Time Limit Exceeded / Runtime Error  
    -1.0 → Compile Error
    ```

- **评估基准**：
  - `MBPP`（Mostly Basic Python Problems）：报告 **Pass@1**
  - `APPS`（Advanced Programming Problem Solver）：按难度分为 Introductory / Interview / Competitive，报告 **Pass@1, Pass@5, Pass@10**

### ⚙️ 实验设置
- **模型家族**：
  - `Qwen Coder` (0.5B, 1.5B, 7B)
  - `DeepSeek Coder` (1.3B, 6.7B)
  - `CodeLlama` (7B)

- **训练配置**：
  - 所有模型均使用 base 版本，**未经过 SFT 或 instruction tuning**
  - 训练轮数：10 epochs
  - Batch size：根据模型大小动态调整（0.5B: 6, 7B: 1–2）
  - Optimizer：RLOO（Leave-One-Out REINFORCE）
  - Advantage Normalization：GRPO 风格归一化
  - Group Design：每组包含至少一个正样本（r=1）和一个负样本（r≤0），以降低 advantage variance

- **硬件资源**：
  - 全部实验在 **单张 80GB A100 GPU** 上完成

### 🔍 基线对比
- 主要对比的是 **各模型自身的 base model**（即未经 RL 微调前的状态）
- 同时与已有工作（如 CodeRL、PPOCoder）进行间接比较（见 Table 1）

---

## 3. **主要实验结果和性能指标**

### 📊 关键性能数据（来自 Figure 1 & Table 3）

| Model | Base Pass@1 (%) | Peak Pass@1 (%) | Improvement |
|-------|------------------|------------------|-----------|
| Qwen 0.5B | 0 | 48 | **+48pp** |
| Qwen 1.5B | 13 | 54 | +41pp |
| Qwen 7B | 64 | 79 | +15pp |
| DeepSeek 1.3B | 56 | 57 | +1pp |
| DeepSeek 6.7B | 3 | 34 | **+31pp** |
| CodeLlama 7B | ~0 | ~9 | +9pp |

> ✅ **结论**：Offline RL 显著提升小到中等规模模型的 zero-shot 代码生成能力，尤其对初始性能差的小模型效果显著。

### 📈 APPS 上的表现（Figure 4 & Figure 8）
- 在更难的 APPS 数据集上也观察到一致提升：
  - Qwen 系列在所有难度级别均有明显增益
  - 小模型（Qwen 0.5B）在 competition-level 也能取得进步
  - CodeLlama 仅在 introductory 级别有效，表明其 pretrain-posttrain 分布存在 **misalignment**

### 🔬 消融实验与关键发现
#### （1）学习率敏感性
- 存在一个“甜点区间”：**1e-5 ~ 5e-5**
  - 过低 → 无改进
  - 过高 → 初期上升快，后期迅速 collapse
- 最佳 LR 因模型而异（非统一）

#### （2）训练 epoch 影响
- 多轮训练有助于性能提升，但 **超过一定 epoch 后出现 model collapse**
  - 如 Qwen 0.5B 在第 4 轮达峰（48%），第 7 轮崩塌至 0%
- Collapse 是普遍现象，存在于所有模型家族

#### （3）不同数据采样影响（Appendix G）
- 使用替换数据集继续训练，性能进一步提升（42 → 46 Pass@1）
- 支持“引入新样本可延缓 collapse”的假设

---

## 4. **关键结论和发现**

### ✅ 主要发现
1. **Offline RL 是可行且高效的**：
   - 仅用几个小时、单张 GPU，即可大幅提升 Code LLM 的 zero-shot 性能。
   - 对 0.5B ~ 7B 的多种架构均有效。

2. **性能提升的关键机制是 logit 分布演化**：
   - 成功训练初期：logit variance 适度增加，模型学会区分好坏样本。
   - 失败训练后期：logit variance 激增（可达数千倍），导致输出退化。

3. **不稳定性的根源不是 advantage variance，而是 logit variance**：
   - Online RL 中 instability 来自 advantage 波动；
   - Offline RL 中 advantage 可控（因 group design），但 **logit gap 扩大引发 collapse**。

4. **Logit Gap 是 collapse 的预警信号**：
   - Negative logit gap 增长过快 → 模型不再从错误样本学习 → collapse
   - 可用于设计 early stopping 准则

5. **Distribution alignment 至关重要**：
   - CodeLlama 表现不佳可能因其 pretraining data 与 CodeNet 差异较大
   - 若 pretrain 和 offline 数据分布接近，则更 likely 成功

### ⚠️ 局限性
- **仅关注 functional correctness**，尚未整合效率、安全性等其他 feedback 类型。
- **未探索多语言场景**：实验仅限于 Python。
- **全量训练（full model training）**，未尝试参数高效微调（如 LoRA）。
- **未验证所有 proposed stabilization 方法**（如 resampling），留待 future work。

### 🔮 未来工作方向
1. 设计 **adaptive early stopping** 策略，基于 logit variance / gap 动态终止训练。
2. 探索 **混合训练范式**：定期从当前 policy 采样新数据加入 offline dataset，缓解 staleness。
3. 扩展至 **multi-objective RL**，同时优化 correctness、efficiency、security。
4. 研究 **distribution alignment 方法**（如数据重加权、领域适配）以提升跨模型泛化性。
5. 应用更大 batch size 和多 GPU 设置，进一步释放潜力。

---

## ✅ 总结一句话
> 本论文证明了 **Offline RL 是一种高效、低成本、高性能的 Code LLM post-training 新范式**，但也揭示了其内在不稳定性源于 **logit variance 扩张**，并提出了诊断与缓解 collapse 的实用方法，为构建下一代可持续、可扩展的代码智能体奠定了基础。

</details>

---

### 12. [CRFCAN: A Complex-Valued Cross-Domain Residual Network for Joint Channel and Phase Noise Estimation in Sub-THz OFDM Systems](https://arxiv.org/abs/2609.12244)

**Authors**: Ruilin Wang, Xiaodai Dong  
**Category**: cs.LG  
**Published**: 2026-09-14  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.12244v1  

#### Abstract
In sub-terahertz (sub-THz) communications, the coupling of ultra-wide bandwidth and severe phase noise (PN) impairments renders conventional joint channel and PN estimation highly complex and computationally prohibitive. To address this, we propose CRFCAN, a complex-valued residual FFT convolutional...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CRFCAN: A Complex-Valued Cross-Domain Residual Network for Joint Channel and Phase Noise Estimation in Sub-THz OFDM Systems

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

在 sub-THz（100–300 GHz）通信系统中，由于超宽带宽和严重相位噪声（**Phase Noise, PN**）的耦合效应，传统的联合信道状态信息（**CSI**）与 PN 估计方法面临以下挑战：

- **计算复杂度高**：迭代算法（如 LMMSE、ECM、EKF）在 sub-THz 大信号维度下计算开销巨大；
- **模型失配**：real-valued neural networks（RVNNs）无法有效建模复数域信号中的相位旋转特性；
- **黑箱设计缺乏物理一致性**：多数深度学习方法为“黑箱”模型，忽略物理约束，泛化能力差；
- **级联架构误差传播**：现有 DL 方法多采用级联网络或混合框架，导致任务间信息割裂。

### **提出了什么新方法或新思路**

作者提出 **CRFCAN**（Complex-Valued Residual FFT Convolutional Attention Network），一种面向 sub-THz OFDM 系统的端到端（end-to-end）复数域残差网络，用于联合估计 CSI 和 PN。

#### 主要创新点：

- ✅ **物理启发的复数域残差学习（Physically Inspired Complex-Valued Rotation Residual Learning）**  
  设计了专用的 **RCAB-PhaseRotation** 残差块（即 PRCAB），将 PN 显式建模为复平面上的**乘性旋转**（multiplicative rotation），而非传统加性扰动。通过嵌入欧拉公式（Euler’s formula），实现细粒度相位补偿。

- ✅ **单次跨域估计架构（Single-Shot Cross-Domain Estimation Architecture）**  
  在残差组中嵌入 **FFT / IFFT 模块**，构建跨时频域的信息交互机制：
  - **频率域**：捕捉频率选择性衰落；
  - **时间域**：跟踪时变相位噪声；
  - 实现单次前向推理完成联合恢复，避免迭代或级联带来的延迟。

- ✅ **端到端联合优化与物理约束输出（End-to-End Joint Optimization with Physical Constraints）**  
  引入**软归一化输出尾部（soft-normalization output tail）** 对 PN 输出施加轻量级幅度约束，使其保持在单位圆附近的一个环形区域内，提升估计稳定性并保留物理特性。

---

### **相比现有方法的优势**

| 维度 | CRFCAN | 传统方法 |
|------|--------|---------|
| **架构形式** | 真正端到端联合估计 | 级联网络 / 混合模型（NN + 迭代算法） |
| **信号表示** | 复数神经网络（CVNN） | 实数网络（RVNN），I/Q 分离处理 |
| **PN 建模方式** | 乘性相位旋转（物理一致） | 加性特征扰动（不匹配） |
| **计算效率** | 单次推理，固定复杂度 | 迭代收敛，延迟不可控 |
| **泛化能力** | 跨 PN 模型无需微调即可泛化 | 依赖特定统计先验，泛化差 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- 数据由 MATLAB 离线生成，基于标准 sub-THz OFDM 系统模型；
- 信道模型：**3GPP TR 38.901 中的 TDL-C 模型**（非视距场景，300 ns 延迟扩展）；
- 相位噪声模型：基于 **3GPP TR 38.803** 的 PLL-based PN 模型，并针对 100 GHz 载波进行校准（△cal = -10 dBc）；
- 测试时评估三种不同 PN 谱（RAN4、RAN1 Set1、Set2），构成跨模型泛化测试。

### **实验设置**

| 参数 | 设置 |
|------|------|
| 波形 | OFDM |
| 子载波数 $N_c$ | 64 |
| OFDM 符号数 $N_t$ | 8 |
| 调制方式 | QPSK, 16-QAM |
| 载波频率 $f_c$ | 100 GHz |
| 脉冲成形滤波器 | Root-Raised Cosine (RRC)，滚降系数 0.25 |
| 信道模型 | 3GPP TDL-C |
| PN 模型 | PLL-based (3GPP TR 38.803) |
| 训练 SNR 范围 | Eb/N0: 20–30 dB |
| 测试 SNR 范围 | Eb/N0: 0–30 dB |
| 批大小 | 128 |
| 优化器 | AdamW |

### **评估指标**

- **NMSE**（Normalized Mean Square Error）：用于评价信道估计精度；
- **MSE**：用于评价 PN 估计误差；
- **BER**（Bit Error Rate）：系统级性能；
- **EVM**（Error Vector Magnitude）：衡量符号畸变程度；
- **参数量 & 推理复杂度**：分析实际部署可行性。

### **基线方法对比**

| 类别 | 方法 | 描述 |
|------|------|------|
| **传统信号处理** | Iterative LS-based Estimator [5] | 基于最小二乘的交替迭代估计，需 block-type pilot |
| **模型辅助学习** | Cascaded Multi-Network [20] | 三级级联网络（ChDNN → PnDNN → DnCNN），依赖初始估计 |
| **全数据驱动接收机** | DeepSRX [22] | 端到端神经收发机，隐式补偿，不显式输出 CSI/PN |

> ⚠️ 所有方法均在同一系统配置下比较，CRFCAN 仅使用 comb-type pilot，而部分基线需额外 block-type pilot，因此其导频开销更低。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ **NMSE 与 MSE 表现（图 7）**

- 在所有三种 PN 模型下，CRFCAN 均显著优于基线方法；
- 随着 Eb/N0 提升，NMSE 和 MSE 同步下降，表明 CSI 与 PN 估计相互增强；
- 在训练域（RAN4）外（Set1/Set2），CRFCAN 仍保持良好性能，无灾难性退化，体现强泛化能力。

#### ✅ **BER 性能（图 8）**

| 方法 | 高 SNR BER 表现 |
|------|----------------|
| **CRFCAN** | 最低 BER，且明显低于 “Perfect H only” 曲线 |
| Perfect H only | 出现明显错误平台（error floor），说明残留 PN 是主要瓶颈 |
| DeepSRX [22] | 性能退化严重，尤其在 Set1/Set2 上 |
| Cascaded [20] | 存在误差传播，性能弱于 CRFCAN |

> 🔥 **关键发现**：CRFCAN 在高 SNR 下 BER 甚至优于“完美信道已知”的情况，证明其提供了有效的**显式 PN 补偿能力**，超越单纯信道均衡。

#### ✅ **EVM 表现（图 9）**

- 随 Eb/N0 增加，EVM 持续降低；
- 16-QAM 的 EVM 平台高于 QPSK，符合预期（对 PN 更敏感）；
- 表明系统性能最终受限于残留 PN/ICI，而非热噪声。

---

### **消融实验结果（Ablation Study）**

在 RAN4 数据集上对关键模块进行移除测试（图 10）：

| 变体 | BER 影响 | 分析 |
|------|----------|------|
| **w/o FFT/IFFT** | 严重恶化，接近最差水平 | 跨域交互是核心，缺失后性能崩塌 |
| **w/o PRCAB** | 明显上升 | 缺少乘性相位建模能力，削弱 PN 补偿效果 |
| **w/o PN Tail** | 轻微上升 | 软约束有助于稳定训练，但非决定性因素 |
| **w/o Uncertainty Weighting** | 性能下降 | 自适应损失平衡对联合优化至关重要 |
| **w/o All Three** | 性能最差，无法超越 Perfect H only | 验证整体设计必要性 |

> ✅ **只有完整 CRFCAN 和 w/o PN Tail 版本能在高 SNR 超越 Perfect H only**，说明：  
> **跨域处理 + 相位感知建模** 是实现显式 PN 补偿的关键组合。

---

## 4. 关键结论和发现

### **主要发现**

1. **跨域结构是解决 CSI-PN 耦合问题的关键**  
   通过在残差组中嵌入 FFT/IFFT，实现了时频域协同推理，有效解耦频率选择性衰落与时变相位噪声。

2. **复数域建模 + 乘性相位旋转机制显著提升 PN 估计准确性**  
   PRCAB 模块通过显式建模相位旋转，使网络行为更贴近物理过程，优于传统加性建模。

3. **端到端联合训练 + 物理约束输出提升鲁棒性和泛化性**  
   软归一化尾部和不确定性加权损失共同提升了训练稳定性，并支持跨 PN 模型零样本迁移。

4. **单次推理架构适合 sub-THz 实时应用**  
   固定复杂度、低延迟，适用于大带宽、高速率系统。

---

### **方法的局限性**

- 当前模型基于 **SISO** 场景，未考虑 MIMO 或 beamforming；
- PN 输出为离散时间序列 $p[n]$，尚未扩展至连续时间轨迹建模；
- 架构复杂度较高（约 4.7M real-valued params），可能限制极低功耗设备部署；
- 依赖 comb-type pilot，若导频稀疏可能导致性能下降。

---

### **未来工作方向**

1. 扩展至 **MIMO-mmWave/sub-THz 系统**，结合波束域信道结构；
2. 引入 **时序建模模块**（如 LSTM、Transformer）以更好捕捉 PN 动态演化；
3. 探索 **无监督/自监督训练范式**，减少对标签数据的依赖；
4. 开发 **硬件友好的轻量化版本**，用于终端芯片集成；
5. 结合 **joint data detection**，构建完整神经接收机 pipeline。

---

## 总结

✅ **CRFCAN 是首个真正实现端到端、跨域、物理一致的 joint CSI-PN 估计网络**，在 sub-THz OFDM 系统中展现出卓越性能：

- 显著优于传统迭代算法与主流 DL 方法；
- 具备单次推理、强泛化、高稳定性等实用优势；
- 为未来 6G 超高速无线通信中的硬件损伤补偿提供了可靠解决方案。

</details>

---

### 13. [Sampling via Decision-Flow: Training-Free Extraction of Improved Latent Reasoning Paths in Large Language Models](https://arxiv.org/abs/2609.12317)

**Authors**: Zhendong Mi, Shaoyi Huang  
**Category**: cs.LG  
**Published**: 2026-09-14  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.12317v1  

#### Abstract
A central question in LLM reasoning is whether reinforcement learning (RL) instills genuinely new capabilities or merely reshapes how existing knowledge is expressed during inference. Building on the distribution-sharpening hypothesis, which holds that RL reallocates probability mass toward high-rew...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Sampling via Decision-Flow: Training-Free Extraction of Improved Latent Reasoning Paths in Large Language Models*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前大型语言模型（LLMs）在复杂推理任务上的表现通常依赖于昂贵的 **Reinforcement Learning (RL)** 后训练（post-training），例如 RLHF 或 RLVR。然而，这类方法存在以下问题：
- **成本高**：需要大量计算资源进行参数更新。
- **多样性下降**：RL 通过“分布锐化”（distribution sharpening）将概率质量集中在少数高奖励路径上，抑制了其他潜在正确但低概率的推理路径。
- **推理覆盖收缩**：尽管峰值性能提升，但模型变得过于自信而缺乏探索能力。

本文提出一个核心科学问题：  
> 如果高质量的推理路径已经以低概率“潜藏”在预训练模型中，是否可以在**不进行任何训练**的情况下将其提取出来？

### 提出的新方法：Decision-Flow Sampling (DF-Sample)
DF-Sample 是一种**无需训练、无需数据**的推理时采样框架，旨在从 base model 中提取高质量但低概率的推理路径。

#### 核心思想
- 不是通过训练引入新能力，而是解决**生成概率与推理质量之间的错配问题**。
- 传统采样策略（如 greedy、beam search）仅基于局部 step-wise 概率决策，容易陷入“看似合理但全局错误”的路径。
- DF-Sample 引入**全局轨迹评估机制**，通过构建推理树并反向传播效用信号，实现对完整推理链的质量感知选择。

#### 方法流程（四阶段）
1. **Hierarchical Reasoning Tree Construction**  
   构建深度为 $L$ 的分层推理树，每个节点扩展 $K$ 个候选推理步骤。
   
2. **Terminal Node Energy Evaluation**  
   定义终端能量函数 $E(\cdot)$ 对叶子节点评分，结合生成似然和输出质量（由 verifier 如 GPT-4o 打分），得分越低表示路径质量越高。

3. **Decision-Flow Backward Propagation**  
   将叶子节点的效用值（utility）沿树向上反向传播，使中间节点能获得其子路径的全局质量估计。

4. **Posterior Path Selection**  
   在每一步选择子节点时，综合考虑原始生成先验 $p_{\text{prior}}$ 和传播来的效用 $U$，形成后验选择策略：
   $$
   \pi^*(v'|v) \propto p_{\text{prior}}(v'|v) \cdot U(v')
   $$

此外，为应对长序列带来的指数级开销，引入 **block-wise sampling** 策略，分块构造与评估推理树。

### 相比现有方法的优势
| 方法类型 | 是否需训练 | 是否利用全局信息 | 是否可恢复低概率高质量路径 |
|--------|-----------|------------------|----------------------------|
| Greedy / Beam Search | ❌ | ❌（局部决策） | ❌ |
| Low-temperature Sampling | ❌ | ❌ | ❌（加剧分布锐化） |
| Power Sampling (MCMC-style) | ❌ | ⭕（局部修正） | ⭕ |
| GRPO (RL-based) | ✅ | ⭕（隐式学习） | ✅（但代价高） |
| **DF-Sample (Ours)** | ❌ | ✅（显式全局评估） | ✅ |

> ✅ **关键优势**：**训练免费 + 显式全局优化 + 能有效挖掘 latent reasoning paths**

---

## 2. 核心实验方法和设置

### 使用的数据集
涵盖数学、编程、科学和通用指令遵循四大领域：

| 数据集 | 描述 | 任务类型 | 评估方式 |
|-------|------|---------|----------|
| **MATH500** | 来自 MATH 数据集的 500 道竞赛级数学题 | 数学推理 | 正确率（Accuracy） |
| **HumanEval** | 164 个手写编程任务 | 代码生成 | Pass@1（单元测试全过） |
| **GPQA-Diamond** | 198 道研究生级别多选题（物理/化学/生物） | 科学推理 | 准确率（最困难子集） |
| **AlpacaEval 2.0** | 805 个开放性指令 | 指令跟随 | GPT-4-Turbo 判断胜率（length-normalized win rate） |

### 实验设置
- **模型家族**：
  - Qwen2.5-Math-7B
  - Qwen2.5-7B
  - Phi-3.5-mini-instruct
- **超参数**：
  - 分支因子 $K=3$
  - 块大小 $B=3$
  - 温度系数 $\alpha=4.0$
- **硬件**：2 × NVIDIA A6000 GPU
- **实现细节**：
  - 终端质量评分 $R(\cdot)$ 使用 GPT-4o 自动打分
  - 所有 baseline 结果来自原论文报告值

### 基线方法对比
| 方法 | 类型 | 特点 |
|-----|------|------|
| **Base Model** | 基线 | 贪心解码（greedy decoding） |
| **Low-temperature Sampling** | 无训练 | 提升输出置信度，强化高概率路径 |
| **GRPO** | RL 训练方法 | Group Relative Policy Optimization，代表性的 RLVR 方法 |
| **Power Sampling** | 无训练 | MCMC 风格重采样，尝试替换低质量段落 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）

| 方法 | MATH500 (Qwen2.5-Math-7B) | HumanEval | GPQA-Diamond | AlpacaEval 2.0 |
|------|----------------------------|-----------|---------------|----------------|
| Base | 0.496 | 0.329 | 0.278 | 1.61 |
| Low-temp | 0.690 | 0.512 | 0.353 | 2.09 |
| GRPO | 0.785 | 0.537 | 0.399 | 2.38 |
| Power Sampling | 0.748 | 0.573 | 0.389 | 2.88 |
| **DF-Sample (Ours)** | **0.818** | **0.591** | **0.456** | **3.06** |

> ✅ **DF-Sample 在所有四个任务上均超越所有 baseline，包括经过 RL 训练的 GRPO**

#### 特别亮点
- 在 **GPQA-Diamond** 上达到 **45.6%**，显著高于 GRPO 的 39.9%，说明其在需要深层科学推理的任务中优势明显。
- 在 **MATH500** 上超过 GRPO **3.3 个百分点**（81.8% vs 78.5%），且无需任何参数更新。
- 在 **AlpacaEval 2.0** 上 win rate 达到 3.06，优于所有方法，表明其泛化至非验证类任务的能力。

### 与其他方法的对比分析

#### Token-Level Confidence Analysis（图7）
- **GRPO**：生成 token 多处于高置信区域 → “过度自信”
- **Power Sampling**：有一定扩散
- **DF-Sample**：**覆盖最广，深入低置信区域** → 成功找到“模型自己都不太信但其实是对的”路径

#### 序列长度与延迟（图8）
- **平均 token 数**：
  - Base: ~600
  - Power Sampling: ~670（最长）
  - **DF-Sample**: ~619（接近 base）
- **延迟**：
  - Power Sampling: ~340s/question
  - DF-Sample: ~384s/question（稍慢但可接受）
- ➜ DF-Sample 更高效地找到了更短但更正确的路径

### 消融实验结果

#### 影响因素：分支因子 $K$（图9）
- 在 GPQA 上测试不同 $K$：
  - $K=2$: 仅比 Power Sampling 高约 1%
  - $K=4$: 高出约 18%
- 结论：更大的搜索空间有助于发现优质路径，但带来更高延迟 → 折中选择 $K=3$

#### 影响因素：温度系数 $\alpha$（表2）
| $\alpha$ | Accuracy (%) |
|--------|--------------|
| 100    | 30.3         |
| 10     | 40.9         |
| 5      | **48.0**     |
| 4      | 45.9         |
| 1      | 27.8         |

- 过大 $\alpha$ → 过度偏向高概率路径 → 忽视潜在优质路径
- 过小 $\alpha$ → 分布太平 → 缺乏区分力
- 最优值在 **$\alpha \approx 4-5$** 之间

#### Pass@k 分析（图10）
- 在 MATH500 上比较不同 $k$ 下的 pass@k：
  - DF-Sample 在 **low-$k$ 区域（如 $k=1,2$）优势最大**
  - 表明其单次采样效率极高，适合资源受限场景
  - 当 $k$ 很大时，各方法趋近，差异缩小

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **高质量推理路径已存在于 base model 中**  
   支持“分布锐化假说”——RL 并未创造新能力，只是重新分配概率质量。

2. ✅ **局部最优 ≠ 全局最优**  
   很多错误路径在每一步都显得“合理”，但整体错误；标准采样易被误导。

3. ✅ **全局轨迹评估至关重要**  
   DF-Sample 通过终端节点评分 + 反向传播，实现了对完整路径的质量感知，突破了局部贪婪限制。

4. ✅ **无需训练也能超越 RL 方法**  
   DF-Sample 在多个 benchmark 上**超越经过 RL 训练的 GRPO**，证明了推理性能提升不一定依赖 fine-tuning。

5. ✅ **成功激活“低概率高价值”路径**  
   实验证明 DF-Sample 能有效采样那些 token-level 置信度低但最终答案正确的路径。

---

### 方法的局限性
- ❗ **纯推理时方法，无法固化知识**  
  所有改进发生在 inference 阶段，模型本身未学习到这些优秀推理模式。
- ❗ **计算开销较高**  
  构造推理树和多次前向推理导致延迟增加（~384s vs ~340s），不适合实时应用。
- ❗ **依赖外部 verifier 打分**  
  终端效用计算依赖 GPT-4o 等强 verifier，在某些领域可能不可行或成本高。

---

### 未来工作方向
- 🔮 **将 DF-Sample 发现的优质路径用于知识蒸馏**  
  通过 distillation 将搜索得到的优质路径“内化”进模型，减少对推理时搜索的依赖。
- 🔮 **结合 RL 与 inference-time search**  
  探索 hybrid 方法：用 DF-Sample 生成高质量数据用于 offline RL 或 SFT。
- 🔮 **设计轻量化版本**  
  降低 block-wise sampling 的冗余计算，提升效率。
- 🔮 **拓展至更多模态与任务**  
  如视觉推理、规划、对话等需要多步推导的任务。

---

> 💡 **一句话总结**：  
> *DF-Sample 证明了“聪明的搜索比昂贵的训练更高效”——只需换个更好的采样方式，就能让 base model 展现出媲美甚至超越 RL 模型的推理能力。*

</details>

---

### 14. [SAGE-Loop: Reliable Closed-Loop LLM-Driven AutoML with Trial-and-Correction and Adaptive Ensembling](https://arxiv.org/abs/2609.12455)

**Authors**: Junquan Gu, Shibo Cui, Xiangfeng Luo, Hang Yu  
**Category**: cs.LG  
**Published**: 2026-09-14  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.12455v1  

#### Abstract
Automated machine learning (AutoML) is reshaping data-driven science and industrial practice, and as large language models are introduced into AutoML, pipeline reliability becomes as important as automation efficiency. However, existing AutoML still struggles to realize instant feedback and adaptive...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SAGE-Loop: Reliable Closed-Loop LLM-Driven AutoML with Trial-and-Correction and Adaptive Ensembling

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

当前主流的 **AutoML** 系统（如 H2O、Auto-sklearn、TPOT）普遍采用“预设搜索空间 → 单次执行 → 固定集成”的**单向流水线范式**，存在以下根本性缺陷：

- **缺乏过程级反馈机制**：一旦某环节失败（如特征无效、模型报错），系统无法进行动态修正，通常直接终止或跳过。
- **生成与使用脱节**：虽然能生成多样化的模型，但集成策略固定（如固定投票），未能基于验证证据动态选择最优聚合方式。
- **可靠性低**：在 LLM 驱动的代码生成中，语法错误、运行时异常频发，缺乏自动修复能力。

这些问题导致 AutoML 在面对复杂任务时表现不稳定，尤其在 LLM 引入后，执行可靠性成为瓶颈。

---

### 🚀 提出了什么新方法或新思路

作者提出 **SAGE-Loop**，一个**可靠的闭环式、自适应的 LLM 驱动 AutoML 框架**，其核心创新在于引入两个关键机制：

#### （1）Trial-and-Correction（试错与修正）
- 构建 **prompt-execution-update** 的轻量级闭环反馈循环。
- 支持多轮 LLM 生成，并结合执行检查、错误日志和验证反馈，实现：
  - **错误驱动修复**（Error-driven Repair）：对报错代码自动注入异常信息并请求重写。
  - **性能驱动修订**（Performance-driven Revision）：对性能不佳的模型引导结构优化。
- 实现“边运行边改进”（improve-while-running）的能力。

#### （2）Adaptive Ensembling（自适应集成）
- 不再使用固定集成模板，而是根据验证证据动态选择集成策略：
  - **监督任务**：在 `stacking` / `bagging` / `voting` 中选择最优方案；若选 stacking，则由 LLM 合成适配当前 base models 的 level-2 learner。
  - **无监督任务**：通过 **co-association matrix + spectral clustering** 融合异构聚类器输出，支持自动估计簇数 $k$ 并过滤退化解。

该框架统一了“如何生成模型”与“如何使用模型”，实现了端到端的可靠自动化。

---

### 🔍 相比现有方法的优势

| 维度 | 传统 AutoML / LLM-Augmented 方法 | SAGE-Loop |
|------|-------------------------------|-----------|
| 执行模式 | One-shot generation，无反馈 | Multi-round trial-and-correction |
| 错误处理 | 失败即终止，需人工干预 | 自动修复与替换 |
| 集成策略 | 固定模板（如固定投票） | 动态选择 + LLM-synthesized meta-learner |
| 模型多样性利用 | 仅用于搜索，未转化为协同增益 | 基于证据的结构互补性利用 |
| 可靠性 | 低，易因代码错误中断 | 高，具备容错与恢复能力 |

---

## 2. 核心实验方法和设置

### 📚 数据集

共使用 **20 个公开数据集**，覆盖三大任务类型：

- **分类任务**（8个）：`cc1`, `ld1`, `credit-g`, `cc2`, `cd2`, `cf1`, `balance-scale`, `jungle_chess`
- **回归任务**（4个）：`boston`, `concrete`, `winequality`, `california`
- **聚类任务**（10个）：复用上述除 `credit-g` 外的数据集（如 `breast`, `glass`, `iris`, `seeds`, `cd2`, `ld1` 等）

主要来自 FinBench 和 UCI/Kaggle 公开库。

---

### ⚙️ 实验设置

- **LLM 配置**：使用 OpenAI API（GPT-3.5-turbo 或 GPT-4o），temperature=0.5，response token limit=200。
- **硬件环境**：AMD Ryzen 7 8845H CPU，32GB RAM，NVIDIA RTX 4060 GPU（8GB VRAM）。
- **每轮设置**：
  - 运行 **R=5 轮** 多轮生成。
  - 每轮最多生成 10 个候选模型，每个模型最多调参 10 次。
  - 特征工程保留 top-10 特征。
- **评估协议**：5 次随机种子运行，报告均值 ± 标准差。

---

### 📊 评估指标

| 任务类型 | 主要指标 |
|--------|---------|
| 分类 | AUC, ACC |
| 回归 | MAE, RMSE, RMSLE（越低越好） |
| 聚类 | ARI（Adjusted Rand Index）, NMI（Normalized Mutual Information） |

---

### 🆚 基线方法对比

| 类别 | 基线方法 |
|------|--------|
| 传统 AutoML | AutoGluon, H2O, TPOT |
| 树模型 | RandomForest, XGBoost, LightGBM |
| LLM-Augmented 方法 | DS-Agent (GPT-4o), CAAFE (GPT-4) |

所有基线均使用默认或推荐超参数，在相同数据划分下公平比较。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）分类任务（Table 1）

- **AUC**：SAGE-Loop 在 8/8 数据集上取得最佳或第二佳成绩。
  - 在 `balance-scale` 上达到 **100.00% AUC 和 ACC**。
  - 在 `jungle_chess` 上 AUC 达 **99.80%**。
- **ACC**：同样全面领先，尤其在 `credit-g` 上显著优于其他方法（79.94% vs 第二名 77.89%）。

> 💡 **优势体现**：相比 CAAFE 和 DS-Agent，SAGE-Loop 更鲁棒，且避免了 OOM 错误（如 CAAFE 在 cd2 上崩溃）。

---

#### （2）回归任务（Table 2）

- 在所有四个数据集上，SAGE-Loop 在 **MAE、RMSE、RMSLE** 上均取得最低值。
  - 例如在 `boston` 上：
    - RMSE: **2.73**（vs XGBoost 2.89，AutoGluon 2.97）
    - MAE: **1.96**（vs 最接近的 2.13）
- 显示出更强的泛化能力和误差控制能力。

---

#### （3）聚类任务（Table 3）

- 在 20 项（10 数据集 × 2 指标）比较中，SAGE-Loop 在 **17 项中排名第一**。
  - 在 `iris` 上 ARI 达 **90.39**（远超第二名 82.64）。
  - 在 `seeds` 上 ARI 达 **87.59**。
  - 在 `glass` 上 ARI 和 NMI 均显著领先。
- 表明其共识聚类机制能有效整合异构输出，提升稳定性。

---

### 🔪 消融实验结果（Ablation Study, Table 4）

移除关键模块后性能下降明显，证明各组件有效性：

| 设置 | 分类（credit-g AUC） | 回归（boston RMSE） | 聚类（seeds ARI） |
|------|---------------------|--------------------|------------------|
| 完整模型（w/all） | **81.08** | **2.73** | **87.59** |
| 移除特征工程（w/o feature） | 80.13 | 2.82 | 64.96 |
| 移除集成（w/o ensemble） | 78.42 | 3.17 | 75.34 |
| 两者都移除（w/o all） | 75.35 | 3.36 | 64.27 |

> ✅ 结论：**自适应集成贡献更大**，但**特征工程与集成协同作用显著增强整体性能**。

---

### 🔄 完成率与可靠性分析（Figure 3）

- **SAGE-Loop** 在 7 个数据集上实现 **5/5 次完整运行成功**（无需人工干预）。
- 对比：
  - DS-Agent：在 `boston` 和 `california` 上为 0/5 成功。
  - CAAFE：多个数据集仅 3/5 成功。
- 移除 correction 模块后，SAGE-Loop 在多个数据集上完成率下降，说明 **trial-and-correction 是保障可靠性的关键**。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **闭环试错机制显著提升可靠性**：
   - SAGE-Loop 能自动检测并修复 LLM 生成中的代码错误和性能不足，实现端到端无人工干预的稳定运行。

2. **自适应集成能有效利用模型多样性**：
   - 不再“为多样性而多样性”，而是将结构差异转化为可验证的性能增益，尤其在 stacking 中通过 LLM 合成定制化 meta-learner 实现精准融合。

3. **统一框架适用于监督与无监督任务**：
   - 提出通用的 adaptive ensemble 设计，在分类/回归中动态选型，在聚类中构建稳定共识，展现出跨任务一致性。

4. **LLM 不仅是生成器，更是推理与修复代理**：
   - 利用 LLM 的上下文理解能力，实现基于错误日志的代码修复和基于性能反馈的模型演进，释放了 LLM 在 AutoML 中的全流程潜力。

---

### ⚠️ 方法的局限性

- **依赖高质量 LLM 接口**：性能受限于所用 LLM 的代码生成与理解能力，本地小模型可能难以复现效果。
- **计算成本较高**：多轮生成 + 多模型训练 + 集成搜索带来更高时间开销，不适合极低延迟场景。
- **提示工程敏感性**：prompt 设计对结果影响较大，需精心设计约束与格式以确保输出可执行性。

---

### 🔮 未来工作方向

1. **扩展至多模态 AutoML**：将 SAGE-Loop 范式应用于图像、文本等非表格数据。
2. **理论分析闭环收敛性**：研究 trial-and-correction 循环是否具有收敛保证及其效率边界。
3. **提升大规模效率**：优化调度策略，减少冗余生成与训练，支持分布式并行。
4. **增强可解释性与可控性**：提供更细粒度的 prompt-level 控制接口，支持用户干预与偏好嵌入。

---

> **总结一句话**：  
> SAGE-Loop 通过引入 **trial-and-correction 闭环机制** 与 **adaptive ensembling**，首次实现了真正意义上的**可靠、自适应、端到端的 LLM 驱动 AutoML**，不仅提升了性能，更解决了执行过程中的稳定性与容错难题，为下一代智能 AutoML 系统提供了新范式。

</details>

---

### 15. [A Differentially Private Federated Proximal Optimization Framework for Customer Churn Prediction in Heterogeneous Federated Telecom Networks](https://arxiv.org/abs/2609.12470)

**Authors**: Joydeb Kumar Sana, Subrata Chakraborty, M M Manjurul Islam  
**Category**: cs.LG  
**Published**: 2026-09-14  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.12470v1  

#### Abstract
Customer churn is one of the major issues in the telecommunication industry. To predict customer churn, conventional centralized machine learning approaches have been widely used. This centralized approach requires customer data to be stored in a central repository, which raises privacy concerns and...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Differentially Private Federated Proximal Optimization Framework for Customer Churn Prediction in Heterogeneous Federated Telecom Networks

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本论文针对电信行业中客户流失预测（customer churn prediction）面临的两大挑战：
- **数据隐私问题**：传统集中式机器学习需要聚合各运营商的原始客户数据，违反 GDPR 等数据保护法规。
- **数据异构性（non-IID）问题**：现实场景中不同客户的使用行为差异大，导致标准 Federated Learning（如 FedAvg）训练不稳定、收敛慢、性能下降。

此外，已有研究在联邦学习中未充分结合 **Differential Privacy (DP)** 和 **统计异质性建模**，缺乏对真实电信网络环境的综合考虑。

### 🚀 提出的新方法
作者提出了一种新的 **DP-FedProx** 框架，将以下两种技术融合：
- **FedProx**：通过引入 proximal regularization（proximal term）增强模型在非独立同分布（non-IID）数据下的鲁棒性和收敛性。
- **Differential Privacy (DP)**：采用 **DP-SGD** 在本地训练过程中添加噪声，防止从梯度更新中推断出敏感信息，提供形式化的隐私保障。

该框架实现了无需共享原始数据、支持多运营商协作、同时应对数据异构与隐私泄露风险的目标。

### 🔍 相比现有方法的优势
| 对比维度 | 本文方法（DP-FedProx） | 现有方法（如 FedAvg / FedProx / 集中式模型） |
|--------|----------------------|---------------------------------------------|
| **隐私保护** | ✅ 引入 DP-SGD，满足 $(\epsilon=1.0, \delta=10^{-5})$ 差分隐私 | ❌ 多数不包含 DP；部分 FL 研究仍需中心化存储数据 |
| **处理 non-IID 数据能力** | ✅ 使用 FedProx 显著提升在异构客户端上的稳定性 | ❌ FedAvg 在非 IID 下表现差 |
| **实用性与合规性** | ✅ 支持跨运营商协作且符合隐私监管要求 | ❌ 中心化方法难以部署于实际业务系统 |

> ✅ **首次将 DP 与 FedProx 结合用于电信客户流失预测任务**，填补了该领域空白。

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
实验基于两个公开的电信客户流失数据集：

| 数据集 | 样本数 | 特征数 | 流失率 | 来源 |
|-------|--------|--------|--------|------|
| **Dataset-1** | 100,000 | 101 | 49.56% | Kaggle（结构类似 Cell2Cell 数据集） |
| **Dataset-2** | 7,043 | 21 | 26.54% | IBM Cognos Analytics 公开样本 |

> ⚠️ 数据预处理包括去标识化、缺失值处理、类别编码，并采用 **Weight-of-Evidence (WoE)** 编码进行特征转换以增强可解释性与模型输入质量。

### 🧪 实验设置
- **客户端划分策略**：按客户 **账户时长（tenure）** 划分为四个 non-IID 客户端：
  - Client 1: ≤12 个月（短生命周期）
  - Client 2: 13–24 个月
  - Client 3: 25–48 个月
  - Client 4: >48 个月（忠诚用户）
  
  此方式模拟真实电信客户群体的自然异质性。

- **通信轮次**：20 轮
- **本地训练**：每轮每个客户端执行 5 个 epoch，batch size = 64
- **优化器**：Adam
- **DP 参数**：$\epsilon = 1.0$, $\delta = 10^{-5}$, clip norm $C = 1.0$
- **模型架构**：统一使用名为 **ChurnNet** 的前馈神经网络（两层隐藏层：128→64→1），含 BatchNorm 与 Dropout

### 📈 评估指标
共使用 **7 个广泛使用的评价指标**：
- Accuracy（准确率）
- Precision（精确率）
- Recall（召回率）
- F1-score（F1 分数）
- Specificity（特异性）
- ROC-AUC（ROC 曲线下面积）
- PR-AUC（PR 曲线下面积）

所有结果均报告全局测试集上的平均性能。

### 🔁 基线方法对比
| 类型 | 方法列表 |
|-----|---------|
| **集中式模型（Centralized）** | Logistic Regression (LR), Random Forest (RF), XGBoost, Gradient Boosting, AdaBoost, ChurnNet |
| **本地模型（Local）** | 各客户端独立训练 LR 模型，最终合并预测 |
| **联邦学习模型（Federated）** | FedAvg, FedProx, DP-FedAvg, **DP-FedProx（本文提出）** |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（取自 Tables 3 & 5）

#### ✅ Dataset-1 结果摘要
| 方法 | Accuracy | F1-score | ROC-AUC |
|------|----------|----------|---------|
| Centralized (LR) | **0.9210** | 0.9159 | **0.9756** |
| Local (LR) | 0.9309 | 0.9262 | 0.9801 |
| FedProx | 0.9195 | 0.9150 | 0.9750 |
| **DP-FedProx（本文）** | **0.9161** | **0.9119** | **0.9735** |

> 💡 **结论**：DP-FedProx 性能接近 FedProx 和最佳集中式模型，仅比 FedProx 下降约 0.34% Accuracy 和 0.31% F1-score，但提供了严格隐私保证。

#### ✅ Dataset-2 结果摘要
| 方法 | Accuracy | F1-score | ROC-AUC |
|------|----------|----------|---------|
| Centralized (LR) | **0.8351** | **0.6638** | **0.8737** |
| Local (LR) | 0.8429 | 0.6774 | 0.8958 |
| FedProx | 0.7903 | 0.5839 | 0.8279 |
| DP-FedAvg | 0.8031 | 0.5956 | 0.8350 |
| **DP-FedProx（本文）** | **0.8003** | **0.5945** | **0.8272** |

> 💡 **结论**：尽管整体性能低于集中式模型（因数据更难、异质性更强），但 **DP-FedProx 表现优于 FedProx**，说明 DP 噪声可能起到正则化作用。

### 🔍 与基线方法的对比结果
| 比较项 | 发现 |
|-------|------|
| **FedProx vs. FedAvg** | 在两个数据集上，FedProx 均显著优于 FedAvg：<br>- Dataset-1: ↑0.26% Accuracy, ↑0.22% F1<br>- Dataset-2: ↑0.78% Accuracy, ↑3.79% F1 |
| **DP-FedAvg vs. FedAvg** | 加入 DP 后性能反而略有提升（尤其 Dataset-2），表明 DP 噪声具有一定的正则化效果 |
| **DP-FedProx vs. FedProx** | 性能下降极小（<0.35%），证明所提方法可在强隐私下保持高可用性 |
| **DP-FedProx vs. Centralized LR** | 虽略低，但在不共享任何原始数据的前提下达到“近似竞争”水平 |

### 🔬 消融实验分析（隐含在比较中）
- **FedProx 成分有效性**：FedProx 明显优于 FedAvg → 验证了 proximal regularization 对 non-IID 的改善作用。
- **DP 成分影响**：DP-FedProx 与 FedProx 差距小 → 表明 DP-SGD 可有效集成而不严重损害性能。
- **组合优势**：DP-FedProx 在 Dataset-2 上甚至超过 FedProx → 暗示 DP 噪声与 FedProx 正则项可能存在协同效应。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **FedProx 显著优于 FedAvg**：在 non-IID 设置下，FedProx 更稳定、收敛更快、性能更高，验证其适用于电信客户异质性强的场景。
2. **DP-FedProx 实现隐私-性能良好平衡**：
   - 在 $\epsilon=1.0$ 的强隐私预算下，性能损失极小；
   - 是首个将 DP 与 FedProx 结合应用于电信客户流失预测的研究。
3. **联邦学习可逼近集中式性能**：在 Dataset-1 上，FedProx 几乎达到集中式 LR 的水平，说明 FL 具备实用价值。
4. **SHAP 分析揭示模型依赖变化**：
   - 集中式模型重视 tenure 和 usage features；
   - DP-FedProx 模型更关注 billing（如 MonthlyCharges）和服务属性（如 InternetService）；
   - 表明 **训练机制会影响 feature importance 排序**，不能直接迁移集中式模型的解释结论。

### ⚠️ 方法的局限性
1. **客户端异质性程度有限**：按 tenure 划分虽合理，但仍属于温和异质性；若客户地域、套餐、文化背景差异更大，效果可能下降。
2. **proximal coefficient $\mu$ 固定为 0.01**：未做自适应调整，可能不是最优配置。
3. **DP 噪声可能掩盖 proximal 效应**：当 DP 添加的噪声量级大于 proximal correction 时，后者的作用被削弱。
4. **仅使用两个数据集**：泛化能力有待更多真实运营商数据验证。

### 🔮 未来工作方向（作者明确指出）
1. **自适应隐私预算分配**（adaptive $\epsilon$ allocation）：根据不同客户端数据质量动态分配隐私成本。
2. **个性化联邦学习**（Personalized FL）：允许客户端保留个性化模型分支，提高局部性能。
3. **Transformer-based 联邦模型**：探索更强大的序列建模能力，捕捉客户行为的时间演化模式。

---

## ✅ 总结
本文提出的 **DP-FedProx 框架**成功地将 **Federated Learning**、**non-IID 优化（FedProx）** 与 **Differential Privacy** 有机结合，为电信行业提供了一个**兼顾高性能与强隐私保护**的客户流失预测解决方案。实验证明其在多个公开数据集上表现优异，且具备良好的可解释性与合规性，是迈向实际部署的重要一步。

</details>

---

### 16. [ForgeMegakernel: A General Framework for Efficient Auto-Regressive Model Decode Megakernels](https://arxiv.org/abs/2609.12379)

**Authors**: Leshan Li, Zhui Zhu, Xianglong Deng, Yaojian Chen, Qingfeng He, Yuxuan Li, Rong Zhao, Xu Han, Zhiyuan Liu  
**Category**: cs.DC  
**Published**: 2026-09-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.12379v1  

#### Abstract
Auto-regressive model decode is bandwidth-bound, since every weight and key/value-cache byte crosses high-bandwidth memory once per token. A megakernel is an ideal solution, but existing automatic megakernel generation approaches cannot achieve both generalization across models and correctness guara...

---

### 17. [RoofLang: Enabling AI-Driven Architecting of LLM Inference Systems](https://arxiv.org/abs/2609.12551)

**Authors**: Ziyue Yang, Yuting Jiang, Lei Qu, Peng Cheng  
**Category**: cs.DC  
**Published**: 2026-09-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.12551v1  

#### Abstract
AI is beginning to make substantive contributions to LLM inference optimization. Existing AI optimizations are predominantly profiling-based. Profiling-bound feedback confines the search to the capabilities and performance of an existing software stack, preventing a fundamentally better architecture...

---

### 18. [LifeMem: Enabling Lifelong Experience Reuse for LLM Agents](https://arxiv.org/abs/2609.12655)

**Authors**: Yuli Qiu, Yutong Li, Wei Su, Zeming Liu, Wanxiang Che, Heyan Huang, Haifeng Wang, Yuang Guo  
**Category**: cs.CL  
**Published**: 2026-09-14  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.12655v1  

#### Abstract
Large language model agents are expected to continuously adapt to new tasks and environments over their lifetime by reusing past experience. However, existing memory-based agents struggle to transfer reusable experience across environments and suffer from catastrophic forgetting as experience accumu...

---

### 19. [Kraken: LLM-based Speech-to-Speech Translation via Low-bitrate VQ and Dual-path Source Conditioning](https://arxiv.org/abs/2609.13045)

**Authors**: Hayato Futami, Hassan Shahmohammadi, Tushar Dhyani, Alkis Koudounas, Rapha\"el Lafargue, Yosuke Kashiwagi, Quentin Jodelet, Emiru Tsunoo  
**Category**: cs.CL  
**Published**: 2026-09-14  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.13045v1  

#### Abstract
Speech-to-speech translation (S2ST) has advanced significantly with speech LLMs, offering the potential for joint optimization and preserving non-linguistic information. However, these models struggle with predicting high-bitrate speech tokens in LLMs, and face the challenge of relying on S2ST train...

---

### 20. [Efficient Vision-Language-Action Management and Serving for Robot Factories](https://arxiv.org/abs/2609.12075)

**Authors**: Dionysios Adamopoulos, Nattapol Chanpaisit, Basel Fakhri, Christina Giannoula  
**Category**: cs.DC  
**Published**: 2026-09-14  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.12075v1  

#### Abstract
Vision-Language-Action (VLA) models show high robotic manipulation capabilities via a two-stage design: a Vision-Language Model (VLM) stage followed by an Action Diffusion Transformer (ADiT) stage. Since robots must meet strict Service-Level Objectives (SLOs) for safety, VLA inference is inherently ...

---

### 21. [HeatCache: Thermal-aware Energy-efficient LLM Inference Scheduling for Chassis-level Liquid Cooling in Sustainable Edge Server Rooms](https://arxiv.org/abs/2609.12449)

**Authors**: Rui Lu, Huanghuang Liang, Kaiqi Guan, Dan Wang  
**Category**: cs.DC  
**Published**: 2026-09-14  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.12449v1  

#### Abstract
LLM inference is increasingly deployed at institution-scale edges to meet service requirements. However, multi-GPU inference consumes a large amount of electricity and produces substantial heat. To improve sustainability, operators and regulations often demand raising the ambient setpoint to reduce ...

---

### 22. [GUIDE: Generative Utility Inference and Decision Engine](https://arxiv.org/abs/2609.12137)

**Authors**: Anagha Tiwari, Alexander G. Gray, Nick Feamster, Brian Jabarian, Alex Imas, Alex Kale  
**Category**: cs.LG  
**Published**: 2026-09-14  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.12137v1  

#### Abstract
Measuring the preferences of human users remains a fundamental challenge of AI alignment. Existing elicitation approaches struggle to efficiently discover multidimensional preferences or accurately ground these inferences in domain knowledge. To address this, we introduce GUIDE, an LLM-driven elicit...

---

### 23. [Repair Before Reinforce: Context-Augmented Knowledge Graph Reasoning for Multi-Hop Question Answering](https://arxiv.org/abs/2609.12230)

**Authors**: Tharaka D. Fonseka, Niraj K. Jha  
**Category**: cs.CL  
**Published**: 2026-09-14  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.12230v1  

#### Abstract
Question-answering often requires reasoning across multiple connected facts rather than retrieving a single isolated relation. Knowledge graphs (KGs) provide a structured way to represent such facts, but training large language models (LLMs) only on isolated KG head-relation-tail triples may limit t...

---

### 24. [FRIST: FMRI Representation Informed Shared-space Training Improves EEG-only Individual-Finger BCI Decoding](https://arxiv.org/abs/2609.12298)

**Authors**: Jintao Zhang, Yidan Ding, Joshua Kosnoff, Maxim Karrenbach, Hanwen Wang, Bin He  
**Category**: cs.LG  
**Published**: 2026-09-14  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.12298v1  

#### Abstract
Finger-level motor decoding is important for naturalistic brain-computer interface (BCI) control, yet individual-finger decoding from scalp electroencephalography (EEG) remains challenging because finger representations are spatially close in the sensorimotor cortex and blurred by volume conduction....

---

### 25. [RiPPLE: Cross-Space Performance Prediction from Early Training for Neural Architecture Search](https://arxiv.org/abs/2609.12418)

**Authors**: Yifan Yang, Zhaoyan Wang, Zheng Gao, Xiaoyu Li, Jiaojiao Jiang  
**Category**: cs.LG  
**Published**: 2026-09-14  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.12418v1  

#### Abstract
Neural architecture search (NAS) evaluates candidate networks, but fully training enough architectures to rank an entire space is expensive. Zero-cost proxies score architectures at initialization, yet their ranking quality varies across search spaces. Learned predictors reduce evaluation cost but t...

---

### 26. [Quality-Constrained Routing over a Fixed Pool of Quantized Mixture-of-Experts Instances](https://arxiv.org/abs/2609.12550)

**Authors**: Zhenghong Huang, Hongfan Wu, Jiheng Zhang  
**Category**: cs.LG  
**Published**: 2026-09-14  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.12550v1  

#### Abstract
Quantized Mixture-of-Experts (MoE) services can hold several pre-materialized instances of one base model, but quantization damage varies sharply across requests and bitwidths. Because instance materialization and replica counts consume memory and require slow reconfiguration, we treat them as upstr...

---

### 27. [MCRL2: Multi-resource Cross-attention-based Representation Learning-augmented Reinforcement Learning for Cloud Microservice Scheduling](https://arxiv.org/abs/2609.13048)

**Authors**: Tiangang Li, Shi Ying, Xiangbo Tian, Chuan Shi, Ding Xiao  
**Category**: cs.LG  
**Published**: 2026-09-14  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.13048v1  

#### Abstract
Efficient microservice scheduling is crucial for maintaining load balance across nodes in data centers and ensuring high quality of service. However, achieving this in practice remains challenging due to dynamic resource imbalance under fluctuating workloads, nonlinear coupling across multiple resou...

---

### 28. [Parameter-Efficient Retrievers for Polish and European Languages](https://arxiv.org/abs/2609.12913)

**Authors**: S{\l}awomir Dadas, Rafa{\l} Po\'swiata, Ma{\l}gorzata Gr\k{e}bowiec, Micha{\l} Pere{\l}kiewicz  
**Category**: cs.CL  
**Published**: 2026-09-14  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.12913v1  

#### Abstract
Dense retrieval systems increasingly rely on multi-billion-parameter language models, whose memory and computational requirements make large-scale indexing, frequent corpus updates, and low-latency serving costly. We present a three-stage training pipeline for developing compact and efficient retrie...

---

### 29. [LLM-Enhanced Dual-Branch Learning for Large-Scale Multi-Label Text Classification](https://arxiv.org/abs/2609.12915)

**Authors**: Hui Ye, Jing Zhang, Xiulong Yang, Rajshekhar Sunderraman  
**Category**: cs.CL  
**Published**: 2026-09-14  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.12915v1  

#### Abstract
Large-scale multi-label text classification assigns a small subset of relevant labels to each document from a vocabulary containing thousands or tens of thousands of candidate labels. Although pretrained language models have improved semantic text representations, most representation-based approache...

---

### 30. [Asynchronous Parallel Search for Exact Multi-Objective Shortest Paths with Versioned Frontier Snapshots and Indexed Dominance Pruning](https://arxiv.org/abs/2609.11944)

**Authors**: Xiaoqing Xu, Ning Zhang, Liuyihui Qian, Xiaojun Liu, Juan Wu, Hong Tang  
**Category**: cs.DC  
**Published**: 2026-09-14  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.11944v1  

#### Abstract
Exact multi-objective shortest-path (MOSP) search computes the complete Pareto set between specified start and goal vertices, and its computational cost can grow rapidly with expanding nondominated label sets and frequent dominance tests over per-vertex Pareto frontiers. Efficiently parallelizing ex...

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

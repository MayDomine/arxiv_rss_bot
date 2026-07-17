# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-07-17 07:41:40 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [PolyQ: Codesigning End-to-End Quantization Framework for Scalable Edge CPU LLM Inference](https://arxiv.org/abs/2607.14618)

**Authors**: Hyunwoo Oh, Suyeon Jang, Hanning Chen, KyungIn Nam, Sanggeon Yun, Ryozo Masukawa, Mohsen Imani  
**Category**: cs.LG  
**Published**: 2026-07-17  
**Score**: 14.5  
**Type**: new  
**ArXiv ID**: 2607.14618v1  

#### Abstract
CPUs are the most universal target for on-device LLM inference, but existing low-bit quantization methods offer either coarse operating points or fine-grained mixed precision that is difficult to execute efficiently on CPUs. We present PolyQ, a CPU-oriented compiler/quantization co-design for activa...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PolyQ: Codesigning End-to-End Quantization Framework for Scalable Edge CPU LLM Inference

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前在边缘设备（如笔记本、嵌入式控制器）上部署大语言模型（LLM）时，**CPU 是最通用且广泛存在的计算平台**。然而，现有的低比特量化方法存在以下问题：

- **粗粒度控制**：传统方法（如 W3/W4）仅支持整数比特配置，难以适应碎片化内存环境下的**分数比特预算**（fractional-bit budget），导致资源浪费或超出限制。
- **执行效率低下**：细粒度混合精度（mixed-precision）虽能提升质量，但在 CPU 上执行困难，因 SIMD 和 LUT 内核要求数据布局规整，不兼容异构比特块。
- **运行时开销高**：通道重排序（reordering）若未在编译期处理，会在运行时引入大量激活值重排（activation reorder）开销。

### 提出的新方法与思路
作者提出 **PolyQ** —— 一种面向 CPU 的端到端量化框架，通过**量化器与编译器协同设计**（quantization-compiler co-design）解决上述问题。

#### 核心创新点：
1. ✅ **可部署的逐通道混合精度量化（Deployable Channel-wise Mixed-Precision Quantization）**
   - 使用一个 **CPU 对齐的比特调色板 {2,3,4,8,16}**，结合 activation-aware saliency 进行 per-channel bit allocation。
   - 支持用户指定任意平均比特预算（如 3.7 b/weight），实现**分数比特控制**。

2. ✅ **编译期布局规整化（Compile-Time Layout Regularization）**
   - 在编译阶段对不规则的混合精度通道进行**排列（permutation）和聚类（clustering）**，形成 bit-homogeneous blocks。
   - 生成专用的 SIMD 或 LUT 兼容内核，并将兼容的排列跨算子传播，避免运行时重排。

3. ✅ **端到端系统级优化验证**
   - 验证了从量化策略到实际推理延迟、吞吐量、能耗的完整链路，证明该方法在真实 CPU 平台上高效可行。

### 相比现有方法的优势
| 维度 | 现有方法（AWQ, Slim-LLM, AMO 等） | PolyQ |
|------|-------------------------------|-------|
| 比特粒度 | 层级或组级整数比特（如 W3/W4） | 逐通道、支持分数比特（如 3.7 b） |
| 执行效率 | 异构比特需运行时转换或无法利用 LUT | 编译期规整为同质块，直接调用高效内核 |
| 运行时开销 | 存在频繁 activation reorder | 通过 DAG 级排列合并，大幅降低重排流量 |
| 能效 | 依赖统一量化路径 | 接近最优 LUT 后端，额外能耗 <2% |

---

## 2. 核心实验方法和设置

### 数据集
- **校准数据集（Calibration Dataset）**：WikiText-2，随机采样 128 条序列用于量化敏感性分析。
- **下游任务评估集**：
  - **WikiText-2**：用于评估 perplexity（越低越好）
  - **MMLU**, **Winogrande**, **ARC-Easy/Challenge**, **HellaSwag**：用于多任务准确率评估

### 实验设置
- **模型**：
  - Falcon-H1-3B
  - Llama2-13B
  - Qwen3-32B
- **目标平台**（代表不同边缘场景）：
  - 工作站级：Ryzen 9 9950X（16核）
  - 笔记本级：Ryzen 7 7840U（8核）
  - 移动级：Intel N250（4核）
- **操作系统与测量工具**：
  - Ubuntu 24.04
  - RAPL 模块测量 energy/token
- **批大小**：batch=1（典型边缘负载）

### 评估指标
| 类别 | 指标 |
|------|------|
| **模型质量** | Perplexity（↓）、下游任务准确率（↑） |
| **资源利用率** | 实际平均比特 `B'`、峰值内存占用 |
| **执行效率** | Prefill latency（↓）、decode throughput（↑） |
| **能效** | Energy per token（J/token） |
| **系统开销** | Activation reorder traffic（MiB/token） |

### 基线方法对比
| 方法 | 特点 |
|------|------|
| **AWQ** | Activation-aware scaling，固定层精度（如 3/4-bit） |
| **GPTQ** | Post-training quantization，支持 W3/W4 |
| **Slim-LLM** | Intra-layer mixed-precision with b±1 palette（含 5–7 bit） |
| **AMO** | Inter-layer mixed-precision，按层分配不同比特 |
| **PolyQ（本文）** | Channel-wise + CPU-aligned palette + compiler co-design |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### ✅ 模型质量显著提升（@3-bit target）
| Model | AWQ | GPTQ | Slim-LLM | AMO | **PolyQ** | **相对提升** |
|-------|-----|------|----------|-----|-----------|-------------|
| Falcon-H1-3B | 5.96 | 6.28 | 6.15 | – | **5.82** | ↑2.4–32.1% |
| Llama2-13B | 5.29 | 5.75 | 5.03 | 5.68 | **4.88** | ↑7.4–28.7% |
| Qwen3-32B | 8.23 | 10.75 | 10.00 | 11.29 | **7.66** | ↑3.0–32.1% |

> 💡 **结论**：在相同平均比特下，PolyQ 显著优于所有基线，尤其在紧约束（如 3b）时优势最大。

#### ✅ 分数比特控制高度精确
- 图 7 显示，在目标比特 $ B \in [3.0, 6.0] $ 区间内，PolyQ 实现的**实际平均比特 $ B' $** 与目标偏差极小：
  - 最大误差仅 **0.045 b**（Falcon-H1-3B）
  - 多数情况下误差 < 0.02 b
- 表明其 **ISA-aware quanta matching** 机制有效保留了预算精度。

#### ✅ 峰值内存占用可控且密集
- 图 8 显示，PolyQ 可提供 **0.1-bit 粒度的内存足迹调节**，而 AWQ 等只能跳跃于 W2/W3/W4。
- 在相同整数比特下，PolyQ 内存开销仅比 AWQ 高 **<1.1%**，说明元数据和混合精度管理开销极低。

#### ✅ 激活重排流量大幅减少
- 图 10：在 B=3.0–4.0 区间，PolyQ 将 activation reorder traffic 降至：
  - **Falcon-H1-3B**: 1.00 MiB/token （vs. Atom: 2.53）
  - **Llama2-13B**: 1.14 MiB/token （vs. Atom: 4.69）
  - **Qwen3-32B**: 2.95 MiB/token （vs. Atom: 8.25）
- **最高减少 70.8%**（Llama2-13B），远超 RPTQ / Atom 等局部融合策略。

#### ✅ 推理延迟与吞吐量可预测缩放
- 图 11：Prefill latency 和 decode throughput 几乎**随目标比特线性变化**：
  - 从 2→3 bit：latency 增加 ~1.56×（接近理论 1.5×）
  - 从 3→4 bit：latency 增加 ~1.38–1.46×（接近 4/3≈1.33）
- 表明 PolyQ 成功将“比特预算”转化为**可预测的设计变量**。

#### ✅ 能耗增加微乎其微
| Platform | AWQ+T-MAC [J/token] | PolyQ [J/token] | **Overhead** |
|---------|---------------------|------------------|--------------|
| Ryzen 9 9950X | 6.03 | 6.10 | **+1.13%** |
| Ryzen 7 7840U | 2.77 | 2.80 | **+0.80%** |
| Intel N250 | 2.37 | 2.41 | **+1.89%** |

> 💡 即使在移动平台上，PolyQ 也仅带来不到 2% 的额外能耗。

---

## 4. 关键结论和发现

### 主要发现
1. 🔍 **分数比特部署在 CPU 上是可行且高效的**  
   通过合理的量化-编译协同设计，可以突破传统整数比特限制，实现平滑的质量-资源权衡。

2. 🧩 **CPU 对齐的比特调色板至关重要**  
   使用 {2,3,4,8,16} 不仅兼容 LUT（2^b 表大小合理），还能保护 outlier channels（用 8/16-bit），同时保持执行效率。

3. ⚙️ **编译期布局规整是降低运行时开销的关键**  
   本地重排序（如 Atom/RPTQ）仍留有大量跨层重排开销；PolyQ 的 DAG 级规划真正实现了“零运行时重排”。

4. 📈 **比特预算成为统一设计变量**  
   同一个参数 $ B $ 可同时控制：
   - 模型质量（perplexity）
   - 峰值内存
   - 推理延迟 / 吞吐
   - 能耗
   形成一条**一致的部署曲线**（deployment curve）。

### 方法的局限性
- ❗ **目前仅支持 weight-only quantization**，未涉及 activation quantization（尽管使用 FP16）。
- ❗ **比特选择受限于预定义集合** {2,3,4,8,16}，无法支持连续比特分配。
- ❗ **依赖静态图结构**，动态分支或多模态输入可能影响排列传播效果。
- ❗ 当前后端绑定 T-MAC/OpenBLAS，移植到其他架构需重新适配 kernel generator。

### 未来工作方向
- ➕ 扩展至 **dynamic quantization** 和 **KV cache compression**，进一步压缩内存。
- ➕ 探索 **更灵活的比特表示**（如 5/6/7-bit with packed SIMD ops）。
- ➕ 支持 **多设备协同推理** 中的分布式 bit allocation。
- ➕ 结合 NAS 或强化学习自动搜索最优 bit map。
- ➕ 将框架扩展至 **GPU 或 NPU** 场景，验证通用性。

---

## 总结
PolyQ 成功弥合了**细粒度量化精度需求**与**CPU 执行效率限制**之间的鸿沟。它不仅提升了模型质量，更重要的是构建了一套**可预测、可控制、高能效**的边缘 LLM 部署体系。其核心思想——**量化与编译协同设计**——为未来低比特 AI 系统提供了重要范式。

</details>

---

### 2. [Scalable Training of Continuous-Time Spiking Neural Networks with Differentiable Spike-Time Discretization](https://arxiv.org/abs/2607.14672)

**Authors**: Yusuke Sakemi, Tomoya Takeuchi, Takeo Hosomi, Kazuyuki Aihara  
**Category**: cs.LG  
**Published**: 2026-07-17  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2607.14672v1  

#### Abstract
Continuous-time spiking neural networks (SNNs) provide an event-driven framework for temporal computation, computational neuroscience, and neuromorphic hardware. However, training deep continuous-time SNNs is severely constrained by the memory required for exact spike-time computation, which evaluat...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Scalable Training of Continuous-Time Spiking Neural Networks with Differentiable Spike-Time Discretization

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
连续时间 Spiking Neural Networks（SNNs）在理论上具有精确的时间编码能力和对类脑硬件的良好映射潜力，但在实际训练中面临以下三大挑战：
1. **内存瓶颈**：基于精确 spike-time 计算的方法需要为每个神经元保留与输入 spike 数量成正比的候选 firing time 区间，导致激活内存消耗高达 $ O(N_{\text{out}} N_{\text{in}}) $，难以扩展到深层网络。
2. **死神经元问题（dead-neuron problem）**：在 TTFS（Time-to-First-Spike）编码下，某些隐藏层神经元可能无法触发 spike，阻断梯度传播，导致训练不稳定。
3. **处理效率低**：深度 SNN 中 spike 逐层传播，导致吞吐率下降，尤其在类脑硬件上难以实现流水线处理。

---

### 🚀 提出的新方法与创新思路

#### （1）**Differentiable Spike-Time Discretization (DSTD)**  
- 将不规则的 presynaptic spike 序列映射为固定时间点上的可微加权事件。
- 引入一个可学习的 kernel 函数 $ K^{\text{DSTD}} $，将每个 spike 映射到相邻两个离散时间点上，从而避免排序操作。
- 该方法适用于任意膜电位和突触时间常数（$ T_u < \infty, T_I < \infty $）的 LIF 模型，并保证在离散时间点上近似膜电位与真实解一致。

> **优势**：
> - 候选区间数量从 $ N_{\text{in}} $ 降为固定的 $ M $（DSTD 步数），内存复杂度降至 $ O(N_{\text{out}} M) $
> - 支持反向传播，兼容自动微分框架
> - 可并行化计算，适合 GPU 加速

#### （2）**Synfire-chain-inspired Temporal Regularization（Syn-SNN）**
- 受 **synfire chain** 动力学启发，强制每一层神经元在预设的时间窗口内 firing。
- 时间窗口随层数逐层平移（shifted by $ T_{\text{shift}} $），形成类似“波”的 spike 传播模式。
- 在损失函数中加入 temporal penalty，鼓励所有神经元 firing 并限制 firing 时间范围。

> **优势**：
> - 缓解死神经元问题
> - 实现 **pipeline-like processing**：前一层完成即可开始处理下一个样本，提升吞吐量
> - 组织化 spike 传播，增强时序信息流动

---

### 🔍 相比现有方法的优势

| 方法类别 | 典型代表 | 局限性 | 本文优势 |
|--------|--------|------|---------|
| Surrogate Gradient Methods (SGM) | [4] | 基于离散时间步，无法控制连续 spike timing；难以映射到模拟硬件 | 支持连续时间建模，更贴近生物机制和硬件实现 |
| Analytical TTFS 方法 | [5–11] | 内存开销大，仅适用于浅层或小型网络 | DSTD 显著降低内存和计算成本，支持大规模训练 |
| Adjoint Methods | [53] | 需求解 ODE，时间步细粒度，效率低 | DSTD 不依赖数值积分，速度快、内存省 |
| 多 spike 分析方法 | [27, 41] | 复杂度高，难扩展 | 本文虽聚焦 TTFS，但 DSTD 可扩展至多 spike 场景（见 Appendix B.4） |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **CIFAR-10**：用于训练 **9-layer convolutional Syn-SNN (Syn-SNN-9)**
- **Fashion-MNIST**：用于训练 **20-layer convolutional Syn-SNN (Syn-SNN-20)**

> 所有图像通过 **TTFS 编码** 转换为 spike 输入：像素强度越高 → spike 发生越早（$ t = T_m(1 - x) $）

---

### ⚙️ 实验设置

| 参数 | 设置 |
|------|------|
| 神经元模型 | Leaky Integrate-and-Fire (LIF)，TTFS 编码 |
| 连接方式 | 卷积层 + 全连接输出层 |
| 架构设计 | 类似 VGG 和 ResNet，引入 **skip-and-delay block**（延迟固定为 $ 2T_{\text{shift}} $）以缓解深层退化 |
| 损失函数 | 分类交叉熵 + temporal penalty + non-firing neuron penalty |
| 优化器 | AdamW，带 warm-up 学习率策略 |
| DSTD 步数 $ M $ | 训练时 $ M=10,20 $；测试时 $ M=40 $（确保精度） |
| 硬件平台 | 单张 NVIDIA GH200 GPU（120GB） |

---

### 📊 评估指标
- **分类准确率（Accuracy）**
- **峰值内存消耗（Peak memory consumption）**
- **单 epoch 训练时间（Training time per epoch）**
- **RMSE of spike times**（与 exact solution 对比）
- **spike-time 分布可视化**：验证 synfire-chain 动态是否形成
- **消融实验**：不同 $ M $、$ T_{\text{width}} $、regularization 权重的影响

---

### 🔁 基线方法对比
- **Exact spike-time computation**：作为主要对比基线，使用完整候选区间计算 firing time
- **其他 continuous-time SNN 方法**：参考文献中的 adjoint、QIF、multi-spike 等（见 Table 8）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）**效率提升显著**
| 指标 | 提升幅度 |
|------|----------|
| **峰值内存消耗** | 最多减少约 **100 倍**（dense LIF 层） |
| **训练速度** | 最多加快约 **20 倍** |
| **可扩展性** | 成功在单 GPU 上训练 **9-layer SNN on CIFAR-10** 和 **20-layer SNN on Fashion-MNIST**

> 图 2(b) 显示当输入 spike 数达 6000 时，内存效率提升 **60–150 倍**

---

#### （2）**模型性能**
| 模型 | 数据集 | 准确率 | DSTD 步数 $ M $ |
|------|--------|--------|------------------|
| Syn-SNN-9 | CIFAR-10 | **90.36%** | $ M=40 $（测试） |
| Syn-SNN-20 | Fashion-MNIST | **92.33%** | $ M=40 $（测试） |

> 准确率接近 exact 方法，且 $ M \geq 20 $ 后趋于饱和（图 3e）

---

#### （3）**近似误差分析**
- 图 7(a)：随着 $ M $ 增加，output layer spike time 的 RMSE 单调下降
- 图 7(b)：$ M=40 $ 时分类准确率已逼近 exact 方法（橙色虚线）

---

#### （4）**消融实验结果**
- **DSTD 步数 $ M $**：
  - $ M=10 $ 已能获得较高准确率
  - $ M=20 $ 基本饱和；过小（如 $ M<15 $）可能导致训练不稳定（尤其 $ T_u=\infty, T_I=T $ 情况）
- **时间窗口宽度 $ T_{\text{width}} $**：
  - 存在 **accuracy vs. latency trade-off**：窗口越宽，准确率越高，但响应越慢
  - 多目标优化（Bayesian optimization）验证该权衡关系（图 3d）
- **Temporal Regularization**：
  - 显著减少 non-firing neurons
  - 成功诱导出跨层的有序 spike 传播（图 3b/c, 4a）

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **DSTD 是一种高效且准确的 spike-time 近似方法**：
   - 在保持高精度的同时，大幅降低内存和计算开销
   - 支持任意 LIF 时间常数配置，具备良好通用性
2. **Synfire-chain regularization 可稳定深层 SNN 训练**：
   - 抑制死神经元现象
   - 诱导出类 synfire chain 的时序动态，促进信息流动
3. **实现 pipeline processing**：
   - 利用时间窗口重置机制，可在当前样本未完全处理完时启动下一输入
   - 显著提升吞吐量，特别有利于深层架构
4. **首次在单 GPU 上实现深层 continuous-time SNN 的端到端训练**：
   - 9-layer on CIFAR-10 和 20-layer on Fashion-MNIST 均成功训练
   - 推动 continuous-time SNN 向实用化迈进

---

### ⚠️ 方法的局限性
1. **仍基于 TTFS 编码**：限制每神经元最多一个 spike，表达能力受限
2. **超参数敏感**：需大量调参（如 $ T_{\text{width}}, T_{\text{shift}}, \gamma_{\text{head/tail}} $ 等），缺乏理论指导
3. **初始化权重分布尚无系统研究**：目前依赖经验设置
4. **DSTD 使用近似梯度**：虽然有效，但其收敛性和泛化性需进一步理论分析

---

### 🔮 未来工作方向
1. **扩展至 multi-spike regime**：结合 Appendix B.4 的思路，发展完整的多 spike DSTD 框架
2. **自动化超参数搜索**：结合 NAS 或元学习，减少人工调参负担
3. **理论分析 DSTD 的梯度偏差与收敛性**
4. **硬件部署验证**：在 analog neuromorphic chips 上实测 energy efficiency 与 throughput
5. **探索更复杂的 temporal coding schema**：如 burst coding、phase coding 等

---

> 💡 **总体评价**：  
> 本文提出的 **DSTD + Syn-SNN** 框架是 continuous-time SNN 可扩展训练的重要突破。它不仅解决了长期存在的内存瓶颈问题，还通过引入生物启发的 synfire-chain 正则化，实现了稳定性与高效性的统一，为 SNN 在机器学习、神经科学和 AI 硬件三个领域的融合应用提供了强有力的技术支撑。

</details>

---

### 3. [Low-Latency Relay Selection in NR-V2X Vehicular Communications via Graph Isomorphism Networks with Edge Features](https://arxiv.org/abs/2607.14176)

**Authors**: Giambattista Amati, Federica Mangiatordi, Emiliano Pallotti, Simone Angelini, Pierpaolo Salvo, Paola Vocca  
**Category**: cs.LG  
**Published**: 2026-07-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2607.14176v1  

#### Abstract
Reliable, low-latency uplink connectivity is a key requirement for C-V2X networks in dense urban environments, where fast channel variations and blockages often degrade direct vehicle-to-infrastructure links. Multi-hop relaying can restore coverage, but relay-link activation under radio, capacity, a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Low-Latency Relay Selection in NR-V2X Vehicular Communications via Graph Isomorphism Networks with Edge Features**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在密集城市环境中，NR-V2X 车联网通信面临严重的信道变化、遮挡和快速拓扑变动，导致车辆到基础设施（V2I）的直接链路可靠性下降。为恢复覆盖，多跳中继（multi-hop relaying）是一种有效手段，但**最优中继选择是一个受无线资源、容量和路由约束耦合的 NP-hard 优化问题**，传统基于 **Mixed-Integer Linear Programming (MILP)** 的求解器计算复杂度高，难以满足低延迟（如 <10ms）的实时性要求。

### ✅ 提出的新方法与创新思路
本文提出了一种 **edge-aware Learning-to-Optimize (L2O) 框架**，结合图神经网络与精确优化，实现低延迟、高性能的中继选择：

- **基于 GINE 的端到端学习模型**：将每个 V2X 快照建模为有向图 $ G=(V,E) $，其中：
  - 节点特征（node features）包含车辆位置、类型、上行流量需求；
  - 边特征（edge features）显式编码无线链路容量（如基于 SNR 的 Shannon 容量）；
  - 使用 **Graph Isomorphism Network with Edge Features (GINE)** 进行消息传递，支持边属性融合，提升对链路质量的感知能力。
- **MILP 作为离线 Oracle 提供监督信号**：通过离线运行 MILP 求解器生成全局最优中继配置（即激活的边集合），用于监督 GINE 模型训练。
- **提出混合策略 GINE-Pruned MILP (GP-MILP)**：
  - 利用 GINE 预测结果对原始图进行**边剪枝**（保留高概率候选边），大幅缩小 MILP 搜索空间；
  - 在剪枝后的图上运行 MILP，既保持解的质量（objective-equivalent），又显著降低求解时间。

### ✅ 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **延迟性能** | GINE 推理延迟稳定在 **5ms 内**，适合实时决策；GP-MILP 平均求解时间低于 30ms，远优于原生 MILP（常超数秒）。 |
| **性能逼近最优** | GINE 在边级预测上与 MILP Oracle 高度一致（F1=0.9544），且最终连通性接近最优。 |
| **可扩展性** | 推理复杂度为 $ O(|E|) $，随图规模线性增长，而 MILP 为指数级，更适合大规模场景。 |
| **架构新颖性** | 是**首批将 MILP 级别最优性与 edge-aware GNN 学习相结合**的工作之一，实现了“学习加速优化”的闭环设计。 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **生成方式**：采用 **OSM-SUMO-GEMV2 联合仿真管道**构建大规模、真实感强的 NR-V2X 数据集。
  - **OSM**：获取罗马 Porta Pia 区域的真实道路拓扑；
  - **SUMO**：模拟车辆移动轨迹，生成动态、异构的交通流；
  - **GEMV2**：基于几何的高效传播模型，符合 3GPP NR-V2X 规范，考虑路径损耗、LoS/NLoS、多径等效应。
- **参数设置**：
  - 载频：5.9 GHz，带宽：800 MHz；
  - RSU 发射功率：10 dBm，车辆：0 dBm；
  - 天线增益：1 dBi；
  - 场景面积：约 1 km² 城市区域；
  - RSU 数量：2~4 个，非均匀部署；
  - 总样本数：**449,500 个图快照**，按 80%/20% 分为训练集与验证集。

### ⚙️ 实验设置
- **模型结构**：
  - GINE 编码器：3 层，每层嵌入维度 256；
  - 边分类头：拼接源节点、目标节点和边特征后输入共享 MLP；
  - 训练：Adam 优化器，学习率 $10^{-5}$，batch size=64，dropout=0.4，共训练 200 轮；
  - 损失函数：**class-weighted binary cross-entropy**，以缓解正负样本不平衡（active/inactive links）。
- **硬件平台**：
  - CPU：Intel i9；
  - GPU：NVIDIA RTX 5000；
  - 测量方式：预热 10 次后取平均推理时间（batch=1），排除数据加载开销。

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy / Precision / Recall / F1-score** | 衡量 GINE 对 MILP 最优边激活决策的预测准确性（边级分类任务） |
| **Connectivity Gain** | 成功连接至至少一个 RSU 的 CAV 占比，衡量端到端通信性能 |
| **Inference Latency** | GINE 前向推理 + 阈值判断的时间 |
| **Solver Runtime** | MILP 或 GP-MILP 的完整求解耗时 |
| **Objective Value** | 是否达到与原始 MILP 相同的最大连通车辆数（用于验证 GP-MILP 的最优性保留） |

### 🔁 基线方法对比
| 方法 | 描述 |
|------|------|
| **MILP Oracle** | 全局最优解，作为“黄金标准”，但计算昂贵，不适用于实时场景 |
| **1-hop MILP Baseline** | 仅允许 CAV 直连 RSU，禁用 CAV-CAV 中继，代表无多跳能力下的上限性能 |
| **GINE-based L2O** | 所提纯学习模型，单次前向推理完成中继选择 |
| **GP-MILP** | 所提混合方法，GINE 预测用于剪枝后再调用 MILP 求解 |

---

## 3. 主要实验结果和性能指标

### ✅ 边级预测性能（vs. MILP Oracle）
在验证集上的表现如下（阈值 T=0.5）：

| 指标 | 数值 |
|------|------|
| Accuracy | **0.9589** |
| Precision | 0.9508 |
| Recall | 0.9581 |
| **F1-score** | **0.9544** |

> 表明 GINE 能高度准确地复现 MILP 的边激活决策。

#### 不同阈值下的权衡（Precision-Recall Trade-off）
| 阈值 | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|-----|
| 0.15 | 0.9251 | 0.8632 | **0.9902** | 0.9224 |
| 0.50 | 0.9589 | 0.9508 | 0.9581 | **0.9544** |
| 0.75 | 0.9552 | **0.9726** | 0.9264 | 0.9489 |

> 可通过调整阈值灵活控制保守性 vs. 连通性保障。

### ✅ 端到端连通性增益（vs. 1-hop MILP）
启用多跳中继带来的平均连通性提升：
- **4 个 RSUs 场景**：**+9.2%**
- **2 个 RSUs 场景（稀疏基础设施）**：**+12%**

> 显示多跳中继在覆盖受限环境下具有显著价值，尤其当 RSU 密度较低时。

### ✅ 推理延迟与可扩展性
| 方法 | <5ms | <10ms | <30ms |
|------|-------|--------|--------|
| **GINE-based L2O** | **100.0%** | **100.0%** | **100.0%** |
| **GP-MILP** | 10.15% | 33.88% | **98.14%** |
| MILP Oracle | 1.1% | 12.44% | 44.87% |

- **GINE 推理时间始终 ≤5ms**，且几乎不受图大小影响（线性复杂度）；
- **GP-MILP 在 >98% 的实例中求解时间 <30ms**，相比原生 MILP 加速两个数量级以上；
- 图越大，MILP 延迟急剧上升，而 GINE 和 GP-MILP 保持稳定。

> **关键发现**：GP-MILP 在几乎所有情况下都能找到与原始 MILP **相同的目标值（objective-equivalent）**，即保持最优性的同时极大提升了效率。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **GINE 可有效学习 MILP 级别的中继决策模式**，在边级预测上达到 **F1=0.9544** 的高精度，说明 GNN 能隐式捕捉复杂的全局约束（如 flow conservation、loop-free routing）。
2. **多跳中继能显著提升城市环境中的连通性**，尤其是在基础设施稀疏区域（+12%），凸显其工程实用价值。
3. **所提 L2O 框架具备极低且稳定的推理延迟（≤5ms）**，完全满足 NR-V2X 对毫秒级响应的需求。
4. **GP-MILP 混合策略实现了“最优性”与“实时性”的统一**：
   - 保留了 MILP 的全局最优解；
   - 将求解时间从秒级压缩至 **<30ms（98%+ 实例）**；
   - 是使精确优化真正可用于现实 NR-V2X 系统的关键一步。

### ⚠️ 方法局限性
- **依赖高质量的离线标签**：需大量 MILP 求解生成训练数据，成本较高；
- **未显式建模时序动态**：当前为静态图建模，未利用历史状态进行序列预测；
- **泛化能力待进一步验证**：测试集中未涉及极端天气、突发事故等非常规场景；
- **中心化假设**：模型假设存在中央控制器收集全局信息，可能限制分布式部署适用性。

### 🔮 未来工作方向
1. 扩展至 **temporal graph modeling**，利用 GNN 处理图序列，增强对动态环境的适应性；
2. 引入 **constraint-aware post-processing** 或 **differentiable optimization layers**，提高预测结果的可行性；
3. 探索 **distributed implementation**，结合 O-RAN 架构，在多个 RSU 或 MEC 节点间协同执行；
4. 在 **multi-service NR-V2X 环境** 下验证方法，支持不同 QoS 要求的服务共存（如安全类 vs. 非安全类业务）。

--- 

> 💡 **总体评价**：该论文成功搭建了 **“学习代理 + 精确优化”** 的桥梁，提出了一套兼具高性能、低延迟和理论保证的中继选择方案，是推动 NR-V2X 向智能化、实时化演进的重要进展。

</details>

---

### 4. [Are LLM-Generated GPU Kernels Production-Ready? A Trace-Driven Benchmark and Optimization Agent](https://arxiv.org/abs/2607.14541)

**Authors**: Lingyun Yang, Yuxiao Wang, Shenghao Liang, Linfeng Yang, Daocheng Ying, Chunbo You, Rui Zhang, Luping Wang, Yinghao Yu, Guodong Yang, Liping Zhang  
**Category**: cs.AI  
**Published**: 2026-07-17  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.14541v1  

#### Abstract
Existing GPU kernel generation benchmarks draw problems from synthetic or curated sources that diverge from deployed workloads. We present Atrex-Bench, a benchmark whose 30 operators and 440 shapes are sampled directly from full-cluster production inference traces of compute-limited, memory-rich GPU...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Are LLM-Generated GPU Kernels Production-Ready?

## 1. 论文的主要贡献和创新点

### 解决的问题
现有 LLM 生成 GPU kernel 的基准测试（如 KernelBench、CUDABench）存在严重偏差：
- **问题来源不真实**：基于合成或人工筛选任务，而非真实生产环境中的负载。
- **评估方式不合理**：对所有算子平等加权，忽略了实际部署中少数关键算子占据绝大部分 GPU 时间的现象。
- **性能天花板模糊**：依赖软件基线而非硬件理论上限（roofline），无法衡量是否真正接近硬件极限。

这导致模型在这些基准上表现良好，但在真实生产环境中仍远未达到可用水平。

### 提出的新方法与创新
本论文提出了两个核心贡献：

#### （1）Atrex-Bench：首个基于全集群生产推理轨迹的 GPU Kernel 生成基准
- **真实负载采样**：从阿里巴巴超 10,000 张 XPU-A 和 H20 加速器上的生产推理服务中提取 trace，覆盖 vLLM、SGLang、AITER、RTP-LLM 等主流框架。
- **重要性加权聚合（Importance-Weighted Aggregate）**：每个算子 `(op, shape)` 的权重由其在生产中消耗的 GPU 时间占比决定，使得评估结果反映真实部署价值。
- **Per-Problem Roofline Ceiling**：为每个问题设定基于硬件峰值计算能力和带宽的理论性能上限，得分定义为 `T_roofline / T_candidate`，确保评估的是逼近硬件极限的能力。
- **隐藏上游信息**：在生成阶段不暴露原始 kernel 名称、实现细节或 roofline 数值，防止模型“作弊”。

#### （2）Atrex-Kernel-Agent (AKA)：一个面向生产的 Profile-Driven Kernel 优化智能体
- **迭代式的 Measure-Revise 搜索流程**：结合官方 profiler（如 ncu）反馈进行多轮优化，每步修改都有证据支持。
- **Optimization Dropout 机制**：当搜索陷入局部最优时，部分重启（mask 掉失败的记忆），保留已接受的 kernel 和审计日志，让新 sub-agent 以更广视角重新探索。
- **分层知识库（Layered Knowledge Base）**：
  - 包含 298 个参考 kernel 文件（NVIDIA/AMD/XPU-Agnostic）
  - 244 份优化知识文档（模式、API/ISA 查阅、陷阱记录等）
  - 可接入上游开源项目（CUTLASS, FlyDSL, AITER 等）用于 API/ISA 查询

### 相比现有方法的优势
| 特性 | 现有基准（如 CUDABench） | Atrex-Bench |
|------|--------------------------|------------|
| 问题来源 | 合成/公开仓库 | **真实生产 trace** |
| 权重分配 | 均匀加权 | **重要性加权（app-card-hour）** |
| 性能上限 | 软件基线 | **Per-Problem Hardware Roofline** |
| 上下文泄露 | 可能暴露原始名称 | **隐藏 provenance 和 roofline** |

AKA 相比传统 one-shot prompting 或简单检索增强，提供了**可审计、可复现、基于实证的优化路径**，解决了“正确性幻觉”和“残余 roofline 差距”两大软失败模式。

---

## 2. 核心实验方法和设置

### 数据集
- **Atrex-Bench Release v1**：
  - **30 个生产级算子**（如 `unified_attention`, `fused_moe`, `block_scaled_mm`）
  - **440 个热点形状（hot shapes）**
  - 来源于 ~20 个生产模型（包括 Qwen3-MoE, DeepSeek-R1, Qwen-VL 等）
  - 覆盖多种精度：`bf16`, `fp8_e4m3`, `fp16`, `fp32`, `int32`

### 实验设置
- **目标 DSL**：FlyDSL（一种在预训练语料中几乎不存在的领域特定语言，避免记忆效应）
- **硬件平台**：主要在 XPU-A（阿里自研加速器）上评测
- **候选模型**：评估了六种前沿编码智能体：
  - Claude Opus 4.7
  - GPT-5.5
  - Qwen3.7-Max
  - Kimi-K2.6
  - GLM-5.1
  - DeepSeek-V4-Pro

### 评估指标
| 指标 | 定义 | 说明 |
|------|------|------|
| **Compile Rate** | 成功编译的 shape 比例 | 第一道关卡 |
| **Correctness** | 编译成功且数值正确的 shape 比例 | 多种子（K=5）、容忍度（atol=1e-2, rtol=5e-2）验证 |
| **FlyDSL Adoption** | 正向传播中在 `@flydsl.kernel` 内部的时间占比 | 衡量是否真正在写目标 DSL，而非 fallback |
| **Roofline Achievement (S_j)** | `T_roofline,j / T_cand,j` | 单个 (op, shape) 单元的性能达成率 |
| **S_agg (Aggregate Score)** | `∑ w_i * S_i` | **核心指标**：按重要性加权的总体 roofline 达成率 |

> 注：`S_agg` 是 operator-balanced 并 importance-weighted，更能反映真实部署收益。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | Compile (%) | Correct (%) | FlyDSL (%) | **S_agg** | vs. torch.compile | vs. Prod Kernel |
|------|-------------|-------------|------------|-----------|------------------|----------------|
| **GPT-5.5** | 100.0 | 91.1 | 71.6 | **0.107** | 3.06× | 0.85× |
| **Claude Opus 4.7** | 99.6 | 92.0 | 78.5 | 0.059 | 2.29× | 0.99× |
| Qwen3.7-Max | 97.1 | 84.8 | 43.8 | 0.047 | 1.10× | 0.19× |
| Kimi-K2.6 | 91.5 | 81.5 | 40.1 | 0.043 | 0.94× | 0.33× |
| GLM-5.1 | 60.9 | 46.2 | 38.6 | 0.015 | 0.97× | 0.33× |
| DeepSeek-V4-Pro | 81.0 | 62.3 | 36.4 | 0.012 | 0.63× | 0.12× |

> ✅ **最佳模型仅达到 ~10.7% 的硬件 roofline**（S_agg = 0.107）

### 与基线方法的对比结果
- **无一模型超越生产内核**：即使是表现最好的 GPT-5.5，平均速度也只有生产内核的 85%。
- **Claude Opus 在正确性上领先，但性能落后**：尽管正确率最高（92.0%），但由于在 compute-bound 算子上表现极差（如 `unified_attention` 上 S=0.01），其 S_agg 远低于 GPT-5.5。
- **FlyDSL Adoption 揭示“正确性幻觉”**：
  - Qwen3.7-Max 正确率达 84.8%，但 FlyDSL 使用率仅 43.8%，说明大量通过 PyTorch fallback 实现。
  - 图 2 显示，多个模型有一半以上的“正确”算子实际上是非 DSL fallback。

### 消融实验与案例研究（AKA 效果）

在三个典型算子上进行了 AKA 控制实验（vanilla vs. AKA-augmented）：

| Model | Operator | FlyDSL (vanilla → AKA) | S (vanilla → AKA) | vs. Prod (vanilla → AKA) |
|-------|---------|------------------------|--------------------|----------------------------|
| Qwen3.7-Max | `chunk_gated_delta_rule_state` | 0% → ~100% | 0.001 → 0.03 | 0.03× → **1.20×** |
| Qwen3.7-Max | `attention_forward` | 0% → 99% | 0.06 → **0.40** | 0.17× → **1.11×** |
| Qwen3.7-Max | `mla_decode_attention` | 14% → 87% | 0.0003 → 0.0035 | 0.10× → **1.06×** |
| Claude Opus 4.7 | `attention_forward` | ~100% → 99% | 0.28 → **0.42** | 0.78× → **1.17×** |

> 🔥 **AKA 成功将 fallback 转换为高性能 FlyDSL kernel，并在多个算子上超越手调生产内核**

---

## 4. 关键结论和发现

### 主要发现
1. **当前 LLM 生成的 GPU kernel 尚未达到生产就绪状态**：
   - 最佳模型也仅能达到 **~10.7% 的硬件 roofline**，远未充分利用硬件能力。
   - “通过测试” ≠ “写出有效 kernel”，存在严重的 **correctness illusion**（靠 PyTorch fallback 通过）。

2. **失败集中在 domain knowledge 缺失，而非编码能力不足**：
   - 写出语法正确的代码容易，但缺乏 **roofline reasoning、tiling 策略、memory layout 优化、dtype-aware fusion** 等专家经验。
   - compute-bound 算子（如 attention, GEMM）是主要瓶颈，而 memory-bound 算子相对容易优化。

3. **AKA 有效填补了这一 gap**：
   - 对弱模型：解决 **target-DSL dominance gap**，将 0% FlyDSL fallback 转为近 100% 的原生 kernel。
   - 对强模型：解决 **residual roofline gap**，进一步提升已有 kernel 的性能，甚至超越 hand-tuned baseline。
   - 证明了 **profile-driven + knowledge-grounded** 的 agentic workflow 是可行路径。

### 局限性
- **Trace Scope 有限**：目前 trace 来自 compute-limited、memory-rich 的推理场景，未覆盖训练、backward kernels 或其他硬件架构（如 NVIDIA GPU）。
- **评估范围较小**：AKA 的消融实验仅在 3 个算子上进行，尚未扩展到整个 Atrex-Bench。
- **依赖高质量知识库**：AKA 的效果高度依赖于 GPU Wiki 中的 reference kernel 和 optimization knowledge 的质量和完整性。

### 未来工作方向
- **跨硬件扩展**：将 Atrex-Bench 扩展至 NVIDIA、TPU 等不同架构，并验证评分体系的通用性。
- **定期刷新基准**：随着生产负载演进，周期性地从新 trace 中更新算子分布和重要性权重。
- **规模化 agentic optimization**：
  - 将 kernel 分解为 prologue/mainloop/epilogue 进行子代理并行调优。
  - 构建 multi-model 协作流程（规划、实现、审查）。
  - 通过长期 campaign（>300 次迭代）反哺 GPU Wiki，形成正循环。

</details>

---

### 5. [Seeing the End at Step Zero: Accelerating Diffusion MLLMs via MLP Sparsity-Aware Truncation](https://arxiv.org/abs/2607.14557)

**Authors**: Qicheng Zhao, Qi Sun, Zheyu Yan  
**Category**: cs.AI  
**Published**: 2026-07-17  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.14557v1  

#### Abstract
Diffusion Multimodal Large Language Models (DMLLMs) are highly effective for multimodal reasoning, yet their inference efficiency is significantly hindered by fixed-length generation constraints. Since the actual output length is unknown, output sequences are padded to a predefined maximum length, r...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《Seeing the End at Step Zero: Accelerating Diffusion MLLMs via MLP Sparsity-Aware Truncation》论文总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对 **Diffusion Multimodal Large Language Models (DMLLMs)** 在推理效率上的一个严重瓶颈——“**填充诅咒 (Curse of Padding)**”——提出了系统性的解决方案。

在 DMLLMs 中，文本生成被建模为在一个预定义的固定长度序列窗口上的全局迭代去噪过程。由于真实输出长度未知，所有样本都会被填充到最大长度，导致大量计算资源浪费在处理无意义的 `[EOS]` 或填充标记上。这种冗余不仅造成计算开销，还会因双向注意力机制引入“噪声泄漏”，损害生成质量。

### 提出的新方法与新思路
论文提出了名为 **Seer** 的训练免费（training-free）加速框架，其核心创新在于：

1.  **Step-0 MLP 激活稀疏性作为语义边界信号**：
    *   **关键发现**：作者发现，在去噪的第一步（Step-0），模型早期层（early layers）的 **MLP 激活值**会呈现出一种独特的稀疏性模式。有效语义前缀（valid prefix）的激活值相对密集，而在预测的 `[EOT]` 标记处会出现一个明显的“**语义跳跃 (Semantic Jump)**”，随后是稳定且高度稀疏的“**填充平台 (Padding Plateau)**”。
    *   这意味着模型在生成开始之前，就已经在内部“知道”了有效输出的结束位置。

2.  **基于信噪比（SNR）的一次性截断（One-Shot Truncation）**：
    *   利用上述发现，Seer 设计了一个 **SNR-aware 边界检测器**，通过分析 MLP 稀疏性曲线中的“跳跃-平台”结构来鲁棒地估计语义边界。
    *   一旦在 Step-0 检测到边界，就立即对所有后续的去噪步骤和网络层执行**一次性宏截断**，永久移除冗余后缀，从而避免了重复计算。

3.  **混合执行路由（Hybrid Execution Routing）**：
    *   为了将算法层面的理论收益转化为实际批处理服务中的吞吐量提升，Seer 设计了一个系统级的执行框架。
    *   它根据动态截断后的序列长度分布，计算“**填充浪费率 (Padding Waste Ratio, PWR)**”，并动态地将序列组分发到三种优化路径：
        *   **Static-Graph Path**：用于长度均匀的组，直接使用高度优化的静态图。
        *   **Bucket-Varlen-Graph Path**：用于长度中等分散的组，进行变长打包以消除填充，同时保持 CUDA Graph 兼容性。
        *   **Eager Path**：用于超短或极不均衡的异常值，使用轻量级执行。
    *   此设计解决了动态截断导致的批处理形状碎片化问题。

4.  **设备驻留延迟阴影（Device-Resident Latency Shadowing）**：
    *   为了避免 CPU-GPU 同步开销，Seer 实现了一个融合的 Triton 内核，在 GPU 设备内存内完成所有截断、重打包和控制信号写入操作，实现了零主机干预的紧凑化。

### 相比现有方法的优势
*   **针对性强**：现有加速方法（如 D3ToM, RedVTP, VisionZip）主要关注**视觉令牌压缩**，而 Seer 专注于解决**文本侧的填充冗余**这一被忽视的瓶颈。
*   **无损且高效**：无需微调（training-free），不损失任何视觉特征，完美兼容 CUDA Graph。
*   **双重收益**：不仅能显著提升效率，还能通过移除冗余后缀来减少“噪声泄漏”，从而在某些复杂任务上**提升准确率**。
*   **系统友好**：其混合执行策略确保了理论上的 FLOPs 减少能有效转化为端到端的墙钟时间加速。

## 2. 核心实验方法和设置

### 使用的数据集
论文在多个广泛使用的多模态理解基准上进行了评估，涵盖了不同领域的任务：
*   **综合推理**：`MME`, `MMMU`, `MMBench`
*   **视觉问答 (VQA)**：`ChartQA`, `ScienceQA (SQA)`, `DocVQA`, `InfoVQA`, `GQA`
*   **数学视觉推理**：`MathVista`

### 实验设置和评估指标
*   **模型**：在三个代表性的 DMLLM 上进行实验：`LaViDa-LLaDA`, `MMaDA`, 和 `LaViDa-Dream`。
*   **评估指标**：
    *   **任务准确率 (Task Accuracy)**：衡量模型的多模态推理能力。
    *   **端到端吞吐量 (End-to-end Throughput)**：单位为 `tokens/s`，衡量推理速度。
    *   **端到端延迟 (End-to-end Latency)**：单位为 `s`，衡量单个请求的响应时间。
*   **硬件**：实验在 NVIDIA A100 GPU 上进行。

### 基线方法对比
论文将 Seer 与多种先进的加速方法进行了比较，主要包括：
*   **扩散模型专用加速**：`D3ToM`, `RedVTP`
*   **自回归模型迁移的视觉令牌压缩**：`VisionZip`, `MMTok`, `DivPrune`, `SparseVLM`
*   **其他基线**：`Focus`, `Daedal` (后者是一种动态每步令牌驱逐方法)

## 3. 主要实验结果和性能指标

### 关键性能数据
*   **吞吐量提升**：Seer 能够将端到端吞吐量提升高达 **~31×**（例如，在 MMaDA 上处理 InfoVQA 时，从 0.13 提升至 4.02 tokens/s）。
*   **延迟降低**：相应地，端到端延迟大幅降低，例如在 LaViDa-LLaDA 上处理 MathVista 时，延迟从 8.79s 降至 0.75s。
*   **准确率保持甚至提升**：Seer 在绝大多数任务上保持了与基线相当的准确率，并在一些复杂的视觉任务上实现了提升。例如，在 `DocVQA` 上，分数从 63.52 提升至 63.66。

### 与基线方法的对比结果
*   **全面领先**：如表1所示，Seer 在**吞吐量和延迟**上全面超越了所有基线方法。
*   **准确率优势**：依赖于视觉令牌压缩的方法（如 D3ToM, RedVTP）通常会因为过度压缩而牺牲准确率。相比之下，Seer 在保持高吞吐的同时，准确率下降更小，甚至有所提升。
*   **系统级优化的有效性**：消融实验证明，仅进行算法截断（Seer Naive Pad）的收益有限，而完整的 Seer 框架（Seer Full）通过混合执行路由，成功将理论收益转化为实际的墙钟时间加速。

### 消融实验结果
*   **超参数影响**：通过调整 `t_jump` 和 `y` 参数，可以显式地在“准确率-吞吐量”帕累托前沿上进行权衡。默认配置 `(t_jump=0.03, y=0.6)` 被证明是理论上的最优解。
*   **`t_pad` 阈值敏感性**：`t_pad` 阈值对性能至关重要。当 `t_pad < 0.7` 时，系统表现稳健；超过此临界点，性能会急剧下降，这证实了混合路由设计的必要性。
*   **边界检测质量**：边界检测的平均绝对误差（MAE）很低（如 DocVQA 上为 0.40），过早截断率（Over-truncation）也很低（如 DocVQA 上为 0.8%），表明检测非常可靠。
*   **系统开销**：Seer 引入的额外开销（边界检测、路由、打包）极小，占总时间不到 0.5%，证明了其高效性。

## 4. 关键结论和发现

### 主要发现
1.  **核心现象**：DMLLMs 在去噪的第一步（Step-0）就通过早期层的 **MLP 激活稀疏性**隐式地揭示了有效语义边界的精确位置。
2.  **双重效益**：移除冗余文本后缀不仅是高效的计算加速手段，也是一种**表示净化过程**。它切断了通过 `[EOS]` 标记进行的“噪声泄漏”路径，使有效文本前缀能更集中地关注正确的视觉目标，从而可能提升推理质量。
3.  **系统协同设计的重要性**：单纯的算法优化不足以实现端到端加速，必须结合像混合执行路由这样的系统级设计，才能将理论收益转化为实践性能。

### 方法的局限性
*   **依赖特定架构**：该方法的有效性依赖于 DMLLMs 的 Transformer 架构及其 MLP 层的行为，可能不适用于所有类型的模型。
*   **极端情况**：对于极短或极长的序列，其检测的鲁棒性需要进一步验证。
*   **动态路由开销**：虽然开销很小，但动态决策和序列重打包本身也引入了一定的管理成本。

### 未来工作方向
*   将 Seer 框架扩展到更广泛的模型架构和应用场景。
*   探索如何将这种“提前知晓终点”的能力应用于其他类型的生成模型。
*   进一步优化混合执行路由策略，以适应更大规模和更多样化的生产环境。

</details>

---

### 6. [Stop Thinking, Start Looking: Efficient Post-Training for Multimodal Document Question Answering via Reasoning-Free Alignment](https://arxiv.org/abs/2607.14682)

**Authors**: Harikrishnan P M, Goutham Vignesh, Ganesh Parab, Saisubramaniam Gopalakrishnan, Vishal Vaddina, Varun V, Rohit Agrawal  
**Category**: cs.AI  
**Published**: 2026-07-17  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.14682v1  

#### Abstract
Efficient multimodal document question answering with explicit visual grounding, locating the precise document region that supports each answer remains an open challenge. Current approaches bifurcate into Supervised Fine-Tuning (SFT), which requires large annotated datasets and reaches optimization ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Stop Thinking, Start Looking: Efficient Post-Training for Multimodal Document Question Answering via Reasoning-Free Alignment*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文聚焦于**多模态文档问答中的视觉定位任务**（Document Visual Grounding, DVG），即在文档图像中不仅生成正确答案，还需精确定位支持该答案的文本区域（bounding box）。当前主流方法存在以下问题：
- **Supervised Fine-Tuning (SFT)** 需要大量标注数据，且优化易饱和（optimization plateau）；
- **基于推理的强化学习**（reasoning-centric RL）依赖冗长的中间思维链（CoT），显著增加推理时的 token 开销，却未带来明确收益。

### 提出的新方法与思路
作者提出 **Perception-RFT** ——一种基于 **Group Relative Policy Optimization (GRPO)** 的后训练框架，其核心思想是：
- **直接感知对齐**（Direct Perception Alignment）：跳过显式推理过程，强制模型将视觉特征直接映射到结构化输出（JSON格式的 `answer` 和 `bbox_2d`）；
- 引入 **Gated Dense Reward 机制**：通过分阶段奖励函数稳定训练，防止模型收敛到“安全但不精确”的大框预测；
- 设计控制变量实验，在相同 RL 框架下比较是否允许生成 `<think>` 推理痕迹，以实证检验“推理是否必要”。

### 相比现有方法的优势
- ✅ **更高效**：推理 token 减少超过 60%，部署成本更低；
- ✅ **更高性能**：在 ID 和 OOD 数据上均优于 SFT 和 reasoning-enabled RL；
- ✅ **更强泛化能力**：尤其在几何定位方面表现出跨域迁移优势；
- ✅ **无需额外标注**：仅需标准 QA + bbox 标注，不依赖人工构造的推理轨迹。

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 类型 | 规模 | 描述 |
|-------|------|------|------|
| **DocILE + FormNLU** | 训练集 | 23,696 样本 | 财务类文档（发票、订单等），用于构建高质量 DVG 语料 |
| **Hold-out Finance Set** | In-Distribution (ID) 测试集 | 6,194 样本 | 同领域财务文档，评估域内精度 |
| **DOGR-Bench** | Out-of-Distribution (OOD) 测试集 | 800 样本 | 包含艺术海报、图表、科学 PDF，视觉结构差异大 |
| **MMDocBench** | OOD 测试集 | 4,028 样本 | 多样化文档类型（报告、收据、信息图等），综合评估泛化性 |

> ⚠️ 所有 OOD 总计 **4,828 个样本**，构成严格的跨域测试基准。

### 实验设置
- **基础模型**：Qwen3-VL-4B（4B 参数规模）
- **硬件平台**：单张 NVIDIA A100 (80GB)，使用 Unsloth 加速 LoRA 微调
- **适配方式**：LoRA (`r=16`, `α=16`)
- **训练流程**：
  - SFT：全量训练 3 轮（约 1,113 步）
  - RFTs：SFT 后接 RL 微调（6k samples，<10% SFT 计算量）
  - RFTb：Cold-start RL（无 SFT 初始化）
  - Reasoning-RFTb：允许生成 `<think>` 推理链的变体

### 评估指标（严格无归一化）
| 指标 | 定义 |
|------|------|
| **F1EM** | 答案字符串级别的 Exact Match F1 分数 |
| **F1loc** | 定位成功定义为 IoU > 0.5 |
| **F1all (Strict Joint Success)** | 必须同时满足 F1EM=1.0 且 IoU ≥ 0.5，模拟企业审计场景 |

所有结果取 3 次独立运行平均值。

### 基线方法对比
| 方法 | 类型 | 是否使用推理链 |
|------|------|----------------|
| Qwen3-VL-4B (Zero-Shot) | 零样本基线 | ❌ |
| SFT | 监督微调 | ❌ |
| RFTs (SFT→RL) | 感知优先 RL | ❌ |
| RFTb (Cold-Start RL) | 冷启动 RL | ❌ |
| Reasoning-RFTb | 允许推理的 RL | ✅ |
| Gemini 3.0 Flash | 商业通用模型（zero-shot） | N/A |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| Model | Setup | F1EM (ID) | F1loc (ID) | **F1all (ID)** | F1all (DOGR) | F1all (MMDoc) |
|--------|--------|-----------|------------|----------------|---------------|----------------|
| Qwen3-VL-4B | Zero-Shot | 0.558 | 0.324 | 0.262 | 0.359 | 0.389 |
| SFT | Supervised | 0.756 | 0.769 | **0.668** | 0.666 | 0.555 |
| **RFTs** | SFT→RL | **0.773** | **0.821** | **0.718** | **0.685** | **0.569** |
| RFTb | Cold-Start | 0.601 | 0.496 | 0.411 | 0.600 | 0.552 |
| Reasoning-RFTb | w/ CoT | 0.550 | 0.395 | 0.303 | 0.382 | — |

> ✅ **RFTs 在所有关键指标上全面超越 SFT 和其他变体**

### 与基线方法的对比结果
- **相比 SFT**：
  - ID 上 F1all 提升 **+5.0 pts**（0.668 → 0.718）
  - OOD 上定位能力大幅提升（如 DOGR: F1loc 从 0.736 → 0.759）
- **相比 Cold-Start RL (RFTb)**：
  - 显著提升稳定性与最终性能，验证了 **SFT 初始化的重要性**
- **相比 Gemini 3.0 Flash**：
  - 尽管参数小得多，但在 ID 金融文档上大幅领先（F1all: 0.718 vs 0.581）
  - 表明 **task-specific alignment 比 model scale 更重要**

### 消融实验结果

#### （1）推理链的影响（Table 2 & Figure 4）
- **Reasoning-RFTb 性能显著低于 RFTb**：
  - ID F1all：0.411 vs 0.303（↓26%）
  - OOD F1all：0.600 vs 0.382（↓36%）
- **训练不稳定**：推理版本出现中期性能下降、后期无法恢复的现象
- **推理长度动态变化**（Figure 5）：
  - 平均推理 token 数从 ~191 压缩至 ~72（**减少 62%**）
  - 表明模型在训练过程中主动压缩甚至消除推理链

#### （2）早期切换到 RL 的效率优势（RFTs-Early）
- 仅用 **300 SFT 步骤**（约 19k 样本曝光）+ 相同 RL 步骤
- 达到与完整 SFT 相当甚至更好的定位性能
- **总训练数据减少 65%**，实现高效学习

#### （3）Grounding Divergence 现象
- 在 OOD 场景下观察到一种新型 trade-off：
  - 几何定位持续提升（如 MMDocBench F1loc ↑）
  - 但语义提取能力可能退化（如 DOGR-Bench F1EM ↓）
- 表明 RL 优化导致 **不对称泛化**：空间模式可迁移，而语义理解受限于训练分布

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **DVG 是一个感知主导的任务**（perception-dominant）：
   - 显式推理链不仅无益，反而损害性能与稳定性；
   - 最优策略趋向于“直觉式”快速响应，而非逐步推导。

2. ✅ **RL 可突破 SFT 的优化瓶颈**：
   - 即使 SFT 已收敛，RL 仍能进一步提升 joint semantic-geometric alignment；
   - 支持“SFT memorizes, RL generalizes”的观点在多模态领域的适用性。

3. ✅ **SFT 初始化至关重要**：
   - Cold-start RL 不稳定，难以达到高性能；
   - SFT 提供稳定的起点，使 RL 能有效探索高阶策略。

4. ✅ **推理链在训练中被自发压缩**：
   - 即便允许生成 `<think>`，模型也会逐渐减少其长度；
   - 最终策略几乎不产生中间推理，说明任务本质不需要 CoT。

5. 🔍 发现新现象：**Grounding Divergence**
   - 在 OOD 下，定位精度提升常伴随语义鲁棒性下降；
   - 提示未来需设计解耦优化目标或引入正则机制。

### 方法的局限性
- 当前研究局限于 **4B 规模模型**，更大模型的行为尚不明确；
- 缺乏带推理标注的公开 DVG 数据集，限制了对 SFT→RL with reasoning 的深入分析；
- Grounding Divergence 尚无有效缓解手段，仍是开放挑战。

### 未来工作方向
- 探索更大规模模型下的 Perception-RFT 可扩展性；
- 构建带有推理标注的 DVG 数据集，推动 hybrid training 研究；
- 设计专门机制来平衡 OOD 下的语义与几何泛化；
- 将 Perception-RFT 应用于其他细粒度视觉定位任务（如表格解析、表单填充等）。

--- 

> 📌 **一句话总结**：  
> *Stop Thinking, Start Looking*——对于文档视觉定位这类感知密集型任务，与其“思考”，不如“看”。Perception-RFT 通过去除冗余推理链、直接对齐视觉特征与结构输出，在更少资源下实现了更高精度与更强泛化。

</details>

---

### 7. [D-cut: Adaptive Verification Depth Pruning for Batched Speculative Decoding](https://arxiv.org/abs/2607.14647)

**Authors**: Tianyu Liu, Yuhao Shen, Rui Cen, Junhan Shi, Jiebin Zhang, Guangshuo Qin, Hong Liu, Song Liu, Guanghua Yu, Jianchen Zhu  
**Category**: cs.CL  
**Published**: 2026-07-17  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.14647v1  

#### Abstract
Speculative decoding accelerates large language model (LLM) inference without compromising output quality. Recent parallel drafting methods further improve single-request performance by decoupling draft length from drafting latency, enabling longer drafts and higher mean accepted tokens (MAT). Howev...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：D-cut: Adaptive Verification Depth Pruning for Batched Speculative Decoding**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在 **Speculative Decoding**（推测解码）中，现有方法（如 DFlash）通过并行生成长草案（long drafts）来提高单请求吞吐量，但在高并发（high concurrency）场景下，这些长草案会显著增加验证成本（verification cost）。由于每个请求都需验证固定长度的草案，而实际接受长度（accepted length）差异很大，导致大量计算资源浪费在被拒绝的草案后缀上，最终使得整体吞吐量下降，甚至低于标准自回归（AR）解码。

### **提出的新方法**
本文提出 **D-cut**，一种**自适应验证深度剪枝策略**，其核心思想是：
- 将整个批次的草案视为共享一个**全局验证预算**（global verification budget）；
- 基于草案模型的**置信度**（confidence）对所有草案位置进行跨请求排序；
- 动态选择最优的剪枝深度，将验证资源集中在最可能被接受的草案上。

### **相比现有方法的优势**
- **跨请求剪枝（Cross-request pruning）**：打破“每请求统一验证深度”的限制，实现细粒度的验证资源分配。
- **运行时自适应（Runtime-adaptive depth selection）**：引入**运行时成本模型**（profiled cost model），根据实际硬件环境（GPU、并行策略等）动态调整剪枝策略。
- **无需训练、无损输出**：不修改目标模型，保留原始输出分布，完全兼容现有系统。
- 在高并发下显著提升吞吐量，恢复甚至超越 AR 解码性能。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **GSM8K**：数学推理任务
- **Math500**：数学问题求解
- **HumanEval**：代码生成
- **MBPP**：Python 编程任务
- **MT-Bench**：多轮对话质量评估

### **实验设置**
- **目标模型**：涵盖密集模型（dense）和 MoE 架构，包括：
  - `Llama-3.1-8B`, `Qwen3-4B/8B/27B`, `Qwen3.5-35B-A3B`, `Hy3-295B-A21B`
- **硬件平台**：
  - 单节点 8× NVIDIA H20（计算受限）
  - 单节点 8× NVIDIA H800（计算富裕）
- **并发级别**：`{4, 8, 16, 32, 64}`
- **草案长度**：D = 15（即验证块大小为 16）
- **实现框架**：基于 vLLM，禁用前缀缓存和异步调度，确保测量聚焦于验证开销。

### **评估指标**
- **Mean Accepted Tokens (MAT)**：每步平均接受的草案 token 数。
- **Throughput Speedup**：相对于标准 AR 解码的输出 token 吞吐量加速比（主要指标）。

### **基线方法对比**
- **DFlash (8)/(16)**：使用 block-parallel 草案模型，分别生成 8 和 16 长度草案。
- **EAGLE-3**：基于动态草案树的方法。
- **MTP**：多 token 预测方法。
- **Standard AR**：标准自回归解码（基准）。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 模型 | 方法 | 平均速度提升（Speedup） |
|------|------|------------------------|
| 所有模型（avg） | DFlash (16) | 1.26× |
| 所有模型（avg） | **D-cut (16)** | **1.65×** |
| MoE 模型（Hy3-295B-A21B） | **D-cut (16)** | **高达 3.0×** |

> ✅ D-cut 在 29/30 个配置中优于 DFlash (16)，平均提速从 1.26× 提升至 1.65×。

### **与基线方法的对比结果**
- **在高并发（bs=64）下**：
  - DFlash (16) 吞吐量饱和甚至**低于 AR 解码**（如在 MT-Bench 上仅 0.48×）。
  - D-cut 显著缓解此问题，在相同条件下仍保持 **1.13×–1.73×** 加速。
- **在 MoE 模型上**：
  - D-cut 实现最高达 **3.0×** 的加速，远超其他方法。
- **在不同数据集上的表现**：
  - 在 **MT-Bench**（短前缀主导）上，D-cut 收益最大，因原方法浪费严重。
  - 在 **Math500/GSM8K**（长接受序列）上，D-cut 仍能有效保留高置信草案。

### **消融实验结果**
#### **（1）剪枝比例的影响（Table 2）**
- 固定剪枝比例（如 0.25, 0.5, 0.75）无法在所有场景下最优：
  - 在 **H20（计算受限）** 上，激进剪枝（0.25）更优。
  - 在 **H800（计算富裕）** 上，保守剪枝（0.5 或 0.75）更好。
- **D-cut (auto)** 动态选择最优比例，在两种设备上均接近最佳固定配置，**无需手动调参**。

#### **（2）采样温度的影响（Figure 5）**
- 当目标模型使用 **temperature=1**（非贪婪采样）时：
  - DFlash 性能进一步恶化（平均仅 0.68× AR）。
  - D-cut 仍维持 **1.15×** 加速，相对 DFlash 提升 **1.68×**。
- 表明：**越复杂的采样模式，越需要自适应剪枝**。

#### **（3）选择器开销分析（Figure 6）**
- D-cut 引入的选择器（selector）仅增加 **0.55–0.58ms** 延迟（占总步长 2–3%）。
- 但通过剪枝，验证阶段延迟降低 **23.7–38.3%**，整体步长缩短 **21–35%**。
- 净收益显著，**开销极低，收益极高**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **验证成本是高并发下的瓶颈**：长草案虽提升 MAT，但验证开销增长更快，导致吞吐下降。
2. **接受长度高度可变**：不同请求、不同步骤间接受长度差异大，固定深度验证效率低下。
3. **D-cut 通过“置信度排序 + 成本建模”实现最优资源分配**：
   - 利用草案模型自身置信度预测接受概率。
   - 结合运行时成本曲线，动态决定剪枝深度。
4. **D-cut 是通用且轻量的调度优化**：无需修改模型结构或训练，即可大幅提升现有 Speculative Decoding 系统效率。

### **方法的局限性**
- **系统集成挑战**：D-cut 产生变长验证批次，与当前主流推理引擎（如 vLLM）的静态 CUDA graph 设计不兼容。
  - 需要支持动态形状捕获或 piecewise graph 复用。
- **依赖高质量置信度信号**：若草案模型置信度不可靠，排序效果下降。
- **目前仅适用于 block-parallel drafters**（如 DFlash），对 tree-based 方法需额外适配。

### **未来工作方向**
- **与推理引擎深度协同设计**：开发支持动态 batch shape 的 CUDA graph 管理机制。
- **扩展到 tree-structured drafters**：将跨请求预算分配应用于 EAGLE、SpecInfer 等树形草案方法。
- **在线学习成本模型**：避免离线 profiling，实现在部署中自动学习验证成本曲线。
- **结合检索增强**：与 Graft 等方法互补，在剪枝后填充高价值候选（如检索结果）。

---

> 🔚 **总结**：  
> **D-cut 提出了一种简单而强大的思想——将验证视为稀缺资源，并通过置信度与运行时代价联合建模，实现动态最优分配。它不仅解决了高并发下 Speculative Decoding 效率退化的问题，还为未来的大规模 LLM 推理系统设计提供了新的调度范式。**

</details>

---

### 8. [DRIFT: Direct Reduced Fourier Transforms for Distributed Spectral Neural Operators](https://arxiv.org/abs/2607.14394)

**Authors**: Sana Taghipour Anvari, David Kaeli  
**Category**: cs.DC  
**Published**: 2026-07-17  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.14394v1  

#### Abstract
Fourier Neural Operators (FNOs) learn solution operators for partial differential equations and offer orders of magnitude speedup over traditional numerical solvers at inference time, which makes them attractive surrogates for high-resolution computational physics. Scaling FNOs to high-resolution sp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**DRIFT: Direct Reduced Fourier Transforms for Distributed Spectral Neural Operators**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在大规模科学计算中，**Fourier Neural Operators (FNOs)** 被广泛用于学习偏微分方程（PDEs）的解算子。然而，在高分辨率、高维空间网格上扩展 FNO 需要将数据分布到多个 GPU 上进行并行计算。传统分布式实现依赖于 **distributed FFT**，其核心问题是：

- 每个 spectral layer 需要多次 **dense all-to-all collectives** 来重分布整个空间张量；
- 这些通信操作传输的是完整的空间域数据（大小为 $N^d$），但随后只保留极小部分频率模式（如 $k_{\text{max}} \ll N$）；
- 因此，**绝大多数通信是冗余且浪费的**，导致通信开销占前向传播时间的 **超过 97%**，成为严重瓶颈。

### 🚀 提出的新方法与核心思想
作者提出 **Distributed Truncated Spectral Transform (DTST)** ——一种全新的通信原语，并基于此构建了 **DRIFT**（Direct Reduced Fourier Transforms）框架。

#### 核心创新：
- **反转操作顺序**：不再“先做 full distributed FFT → 再 truncation”，而是 **直接在本地计算所需的频谱系数（partial DFT）→ 再通过全局聚合得到完整保留谱**。
- 利用 DFT 的线性性质，每个 GPU 只在其局部空间分区上计算目标频率模式的 **partial spectrum**，然后通过 **AllReduce** 合并所有 GPU 的贡献。
- 最终仅对保留的 $M = (2k_{\text{max}})^d$ 个模式进行通信，而非整个 $N^d$ 张量。

#### DRIFT 架构特点：
- 使用可分离的 per-dimension partial DFT，通过预计算的 **basis matrix** 实现；
- 利用 **cuBLAS** 和 GPU-aware MPI 实现高效矩阵乘法与通信；
- 支持 **spatial data parallelism** 与 **spectral weight model parallelism** 的统一。

### 🔍 相比现有方法的优势
| 方面 | 传统 DFNO（基于 distributed FFT） | DRIFT |
|------|-------------------------------|--------|
| 通信量 | $O(N^d / P)$ per all-to-all | $O(M)$，独立于空间分辨率 |
| 通信次数 | 每层 4 次 all-to-all（repartition） | 每层仅 2 次 collective（AllReduce + AllGather） |
| 通信占比 | >97% 前向时间 | <6% 前向时间 |
| 计算效率 | 即使丢弃大部分系数仍需执行 full FFT | 仅计算所需模式，避免无用计算与内存写入 |
| 扩展性 | 通信随 $P$ 线性增长（$O(P)$ latency） | 通信延迟为 $O(\log P)$，更具可扩展性 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- 使用 **PDEBench** 中的 **3D compressible Navier-Stokes dataset**：
  - 输入维度：$(128, 128, 128, 21, 5)$，即 128³ 空间网格、21 时间步、5 场变量（密度、速度、压力等）；
  - Mach 数为 1.0，近无粘流 regime；
  - 训练/测试划分：90/10。

### ⚙️ 实验设置
- **模型配置**：
  - FNO 结构：4 个 Fourier blocks；
  - 通道宽度 $d_v = 20$；
  - 保留模式数 $(k_x, k_y, k_z, k_t) = (8, 8, 8, 16)$；
  - 输入时间步 $t_{\text{in}} = 5$，预测输出 $t_{\text{out}} = 16$。
- **硬件平台**：
  - GPU：NVIDIA Tesla V100-SXM2-32GB；
  - 拓扑：每节点 4 GPU（NVLink），共 1–8 节点（总计 4–32 GPUs）；
  - 网络：InfiniBand EDR（100 Gb/s）；
  - 通信库：HPC-X（OpenMPI），支持 GPU-aware MPI。
- **并行策略**：
  - 空间分解采用 **1D slab decomposition**（沿 x 维度）；
  - 当局部 x 维度过小时切换至 **2D decomposition**（$P_x=16, P_y=2$）；
  - 每个 GPU 对应一个 MPI rank。

### 🎯 评估指标
- **前向传播时间（forward-pass time）**
- **反向传播时间（backward-pass time）**
- **训练总耗时（wall-clock time per epoch / total training time）**
- **通信时间占比**
- **相对误差（Relative Frobenius Error, L2 Error）**
- **收敛曲线（loss vs. epoch / vs. wall-clock time）**
- **强扩展性（strong scaling）与弱扩展性（weak scaling）表现**

### 🔁 基线方法对比
- **Baseline**: 分布式 FNO（DFNO）[Grady et al., 2023]，使用标准 distributed FFT + truncation；
- **对比项**：
  - 通信开销（collective time）
  - 总运行时间
  - 可扩展性趋势
  - 数值精度一致性

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### ✅ 前向传播加速比（Forward Pass Speedup）
| 网格大小 | GPU 数量 | DRIFT 前向时间 (ms) | DFNO 前向时间 (ms) | 加速比 |
|---------|----------|---------------------|--------------------|-------|
| 128³    | 4        | 148.3               | 9443.1             | **63.7×** |
| 128³    | 8        | 87.7                | 5154.8             | **58.8×** |
| 128³    | 16       | 56.5                | 3202.6             | **56.7×** |
| 128³    | 32       | 43.4                | 1634.1             | **37.7×** |

> 在 128³ 网格下，**平均前向加速达 38–64×**。

#### ✅ 训练端到端加速
- 在 **P=16 GPUs** 上训练 100 epochs：
  - **DFNO**: 耗时 **6.8 小时**
  - **DRIFT**: 耗时 **11 分钟**
  - ➜ **37× wall-clock training speedup**

#### ✅ 通信时间显著降低
- DFNO 中通信占前向时间 **>97%**；
- DRIFT 中通信仅占 **<6%**；
- 例如在 32 GPUs 下：
  - DFNO collective time: **891 ms**
  - DRIFT collective time: **~20 ms**（>44× 减少）

#### ✅ 弱扩展性表现
- 局部负载固定（每 GPU 处理 $4\times128\times128$）；
- 全局网格从 $16\times128^2$ 扩展到 $128^3$；
- DRIFT 前向时间仅从 **28ms → 43ms**（+1.5×），而 DFNO 从 **810ms → 1634ms**（+2×）；
- 最大前向加速比达 **43×**（P=16），整体维持在 **24–38×**。

### 🔍 消融实验与敏感性分析

#### 🔤 数值精确性验证
- 图 7 显示 DRIFT 与 DFNO 输出的 **bit-wise identical**（相对 L2 error = 0）；
- 保留频谱系数的 Frobenius error 仅为 **3.2×10⁻¹⁴**，达到双精度浮点极限；
- 表明 DTST 是 **exact** 的数学等价变换。

#### 📈 不同 $k_{\text{max}}$ 的影响（图 11）
- 随着保留模式增加（$k_{\text{max}}$ ↑），DRIFT 通信负载上升，优势略有下降；
- 但在典型范围 $k_{\text{max}}=4{-}16$ 内，仍保持 **18–69×** 加速；
- 特别是在低 $k_{\text{max}}$ 或高分辨率场景下，优势更明显。

#### 💡 每阶段耗时分解（图 9）
- 在 128³、P=4 时：计算主导（partial DFT 占 44%）；
- 在 P=32 时：通信（AllReduce + AllGather）成为主导（但仍 <20ms）；
- 表明 DRIFT 成功将瓶颈从通信转移到可并行化的本地计算。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **通信冗余是分布式 FNO 的根本瓶颈**，尤其是在高分辨率、多维设置中；
2. **DTST 逆转了“FFT → truncation”流程**，实现了真正的 **communication-avoiding spectral transform**；
3. DRIFT 在不牺牲数值精度的前提下，将通信体积从 $O(N^d)$ 降至 $O(k^d)$，且通信延迟为 $O(\log P)$；
4. 在真实 PDE 任务上实现了 **38–64× 前向加速** 和 **37× 端到端训练加速**；
5. 随着问题规模增大（分辨率↑、维度↑），性能增益持续扩大。

### ⚠️ 方法的局限性
- **当 $k_{\text{max}}$ 接近 $N$ 时，partial DFT 的计算成本可能超过 cuFFT**（文中指出 crossover 点约在 $k_{\text{max}} \sim 10$）；
- DRIFT 的通信负载虽小但固定（$O(M)$），在极高 GPU 数量下可能再次成为瓶颈；
- 当前评估集中在特定 FNO 架构，其他 spectral neural operator（如 AFNO、TurboFNO）上的泛化效果有待验证；
- 实际网络拓扑与拥塞未被建模，超大规模部署可能存在偏差。

### 🔮 未来工作方向
- 将 DTST 推广至其他 **truncated spectral methods**，如：
  - Spherical Harmonic Transforms（气候模拟）
  - Spectral Element Methods
  - Vision Transformers 中的低频注意力机制
- 结合 **adaptive mode selection** 动态调整 $S$ 集合；
- 探索 **混合精度 + 压缩通信** 进一步优化 AllReduce 开销；
- 在千卡级系统上验证其 **超大规模扩展潜力**。

---

> **总结一句话**：  
> DRIFT 通过重新思考分布式频谱变换的本质，用 **local partial DFT + global AllReduce** 替代传统的 **distributed FFT + truncation**，从根本上消除了通信冗余，在保持数学等价性的前提下实现了数量级的性能飞跃，为大规模科学机器学习提供了高效的分布式基础构件。

</details>

---

### 9. [A Continuous-Time Reinforcement Learning Framework for Fine-Tuning Discrete Diffusion Models](https://arxiv.org/abs/2607.14522)

**Authors**: Zikun Zhang, Jiayuan Sheng, David D. Yao, Wenpin Tang  
**Category**: cs.LG  
**Published**: 2026-07-17  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.14522v1  

#### Abstract
We formulate reinforcement learning (RL) in continuous time with discrete state spaces and possibly arbitrary action spaces via a stochastic control approach, where the state dynamics are modeled as a controlled continuous-time Markov chain (CTMC). We consider policy optimization problems and derive...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Continuous-Time Reinforcement Learning Framework for Fine-Tuning Discrete Diffusion Models

---

## 1. 论文的主要贡献和创新点

### 解决的问题
- **离散扩散模型（discrete diffusion models）的 fine-tuning 难题**：传统基于 RL 的 fine-tuning 方法在离散空间中面临采样不可导（non-differentiable sampling）、轨迹似然（trajectory likelihood）难以计算等问题，尤其在任意顺序解码（any-order decoding）的 dLLMs 中更为严重。
- **现有 GRPO 方法的局限性**：大多数方法仅依赖终端奖励（terminal rewards），缺乏对中间去噪步骤的有效信用分配（credit assignment），且似然估计粗糙、理论基础薄弱。

### 提出的新方法与新思路
- **提出 Continuous-Time Reinforcement Learning (CTRL)** 框架：
  - 将离散扩散过程建模为受控的连续时间马尔可夫链（controlled CTMC），将 denoiser 的概率向量视为 action，score function 视为 policy。
  - 在连续时间下推导出 policy gradient、HJB 方程、q-function 和 PPO/GRPO 算法变体，实现对任意顺序生成过程的 principled RL 优化。
- **支持中间奖励（intermediate rewards）**：
  - 允许在整个 denoising 轨迹中引入过程监督（process supervision），通过中间状态的 reward 或 advantage 信号指导训练，提升信用分配精度。
- **统一视角下的策略参数化设计**：
  - 对于 masked diffusion models (MDMs)，提出多种策略参数化方式（Dirichlet, temperature softmax, logistic-normal），并分析其概率比（probability ratio）的解析形式。
- **轨迹子采样（trajectory subsampling）技术**：
  - 为降低在线训练中的前向传播次数，提出从完整去噪轨迹中随机采样子集进行梯度更新，显著减少计算开销，同时保持无偏估计。

### 相比现有方法的优势
| 维度 | 本方法 (CTRL) | 现有方法（如 d1, d2, SPG） |
|------|----------------|----------------------------|
| **理论基础** | 建立在连续时间随机控制之上，具有坚实的数学推导 | 多为启发式似然估计，缺乏统一 MDP 框架 |
| **奖励机制** | 支持中间 reward，实现细粒度过程监督 | 主要依赖最终完成结果的终端 reward |
| **策略灵活性** | 支持任意 action space 和 policy parameterization | 多采用固定形式的似然近似 |
| **计算效率** | 引入 trajectory subsampling，大幅减少 forward passes | d2 等需多步 forward；d1 使用均场近似但精度较低 |

---

## 2. 核心实验方法和设置

### 数据集
- **低维合成任务**：
  - 二维 90×90 离散网格上的 checkerboard 分布匹配任务，用于验证 PPO 在熵正则化目标下的收敛性。
- **真实世界推理与编码任务**：
  - **数学推理**：
    - `GSM8K`：小学数学应用题
    - `MATH500`：高中竞赛级数学题
    - `Countdown`：组合算术游戏
    - `Sudoku`：4×4 数独求解
  - **代码生成**：
    - `HumanEval` 和 `MBPP`：Python 编程任务
  - **基础模型**：LLaDA-8B-Instruct（8B 参数的 diffusion-based LLM）

### 实验设置与评估指标
- **训练设置**：
  - 使用 GRPO 算法进行 online RL fine-tuning。
  - completion length 设置为 128 和 256。
  - 每个任务独立训练，共 1000 步。
  - 使用 LoRA (rank=128, α=64) 进行参数高效微调。
- **评估方式**：
  - 所有评估使用 **0-shot prompting**（Sudoku 除外，SPG 使用 3-shot）。
  - 使用 d1 官方评估脚本，batch size=8。
  - 报告每个任务的最佳 checkpoint 性能。
- **中间奖励设计（IRF）**：
  - 设计 mask-tolerant 的可验证中间奖励函数，例如：
    - GSM8K：答案区域合法性惩罚、尾部内容惩罚、标签结构错误惩罚。
    - Sudoku：部分填充格子的合法性检查与违规比例打分。

### 基线方法对比
- **d1 (Zhao et al., 2025c)**：首个将 GRPO 应用于 dLLMs 的方法，使用均场近似估计轨迹似然。
- **d2 (Wang et al., 2026c)**：基于块复合似然（block composite likelihood）改进似然估计。
- **SPG (Wang et al., 2026b)**：使用 sandwiched policy gradient 框架，结合 ELBO/EUBO 边界估计。
- **LLaDA-8B-Instruct（未微调）**：原始 base model。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & 2）

#### 数学推理任务（Table 1）
| Model / Seq Len | GSM8K (128) | GSM8K (256) | MATH500 (128) | MATH500 (256) | Countdown (128) | Sudoku (128) |
|------------------|-------------|-------------|---------------|---------------|------------------|--------------|
| LLaDA-8B-Instruct | 67.9        | 76.1        | 26.8          | 33.4          | 21.2             | 11.5         |
| d1               | 70.9        | 76.6        | 28.2          | 36.6          | 28.4             | 22.9         |
| d2               | 74.8        | 78.9        | 32.2          | 37.8          | 47.8             | 29.9         |
| SPG              | 75.7        | 78.1        | 32.0          | 37.8          | 64.9             | 26.1         |
| **CTRL (ours)**  | **74.3**    | **80.3**    | **32.6**      | **38.2**      | **58.7**         | **88.2**     |

> ✅ **Sudoku 上取得压倒性优势（88.2 vs 第二名 29.9）**

#### 编程任务（Table 2）
| Model / Seq Len | HumanEval (128) | HumanEval (256) | MBPP (128) | MBPP (256) |
|------------------|------------------|------------------|------------|-----------|
| LLaDA-8B-Instruct | 24.8            | 34.8            | 39.7       | 40.5      |
| d1               | 29.3             | 39.0             | 42.0       | 45.5      |
| d2               | 39.6             | 48.7             | 45.6       | 46.8      |
| SPG              | 29.3             | 40.2             | 44.4       | 44.8      |
| **CTRL (ours)**  | **62.8**         | **66.2**         | **52.9**   | **57.7**  |

> ✅ **在 HumanEval 和 MBPP 上全面领先，最高提升超 20 个百分点**

### 消融实验结果
- **轨迹子采样大小 $N$ 的影响（Figure 3）**：
  - 在 GSM8K 上测试不同 $N$（每轮更新使用的去噪步数）：
    - $N=2$: 收敛不稳定
    - $N=4$: 开始稳定
    - $N=8$: 已接近 $N=16$ 和 $N=32$ 的最终性能
  - **结论**：$N=8$ 是性能与效率之间的最佳平衡点，相比完整轨迹 ($T≈128$) 减少了约 16 倍的 forward passes。
- **温度参数 $\lambda$ 的影响（Figure 4）**：
  - 在指数分布 $T_j \sim \text{Exp}(\lambda)$ 控制探索强度时：
    - $\lambda=2.0$ 表现最优
    - $\lambda$ 过小或过大都会导致过度探索，损害生成质量

---

## 4. 关键结论和发现

### 主要发现
1. **CTRL 框架是 principled 且通用的**：
   - 成功将 continuous-time RL 理论扩展到 discrete state spaces，填补了该领域的空白。
   - 为 fine-tuning 任意 score-based 离散扩散模型提供了统一框架。
2. **中间奖励至关重要**：
   - 特别是在 Sudoku 等需要系统性推理的任务中，过程监督显著提升了模型逐步构建正确解的能力（见 Figure 5，中间 reward 快速上升至 0.9）。
3. **轨迹子采样有效缓解计算瓶颈**：
   - 提供了一种简单而高效的解决方案，在不牺牲理论性质的前提下大幅提升训练吞吐量。
4. **在多个复杂任务上实现 SOTA 性能**：
   - 在数学推理和代码生成任务上全面超越 d1、d2、SPG 等先进方法，尤其在 Sudoku 上表现惊人。

### 方法的局限性
- **高方差估计风险**：尽管 trajectory subsampling 是无偏的，但在 $N$ 很小时可能导致训练不稳定。
- **PPO 实现仍具挑战**：文中指出尝试用 rollout 估计 value function 导致高方差，目前更推荐使用 GRPO（critic-free）。
- **大规模部署成本**：虽然比全轨迹计算轻量，但仍需多次 forward pass，对于超大模型仍有负担。

### 未来工作方向
- 开发更高效的 importance sampling 或 variance reduction 技术以进一步减少 function evaluations。
- 探索 PPO 在 dLLMs 上的稳定训练方案，例如更好的 value function approximation。
- 将 CTRL 框架推广至其他非自回归生成任务，如图像编辑、分子设计等。
- 结合 offline RL 范式，构建 hybrid training pipeline。

--- 

> **总结**：本文提出了一个理论严谨、实践有效的 continuous-time RL 框架（CTRL），首次实现了对离散扩散模型的 principled fine-tuning。通过引入中间奖励和轨迹子采样，不仅解决了现有方法的关键缺陷，还在多个高难度任务上取得了显著性能突破，为 diffusion-based LLMs 的强化学习后训练开辟了新路径。

</details>

---

### 10. [CausalGraphX: A Counterfactual Graph Neural Network Framework for Explainable Systemic Risk Assessment](https://arxiv.org/abs/2607.14416)

**Authors**: Rabimba Karanjai, Hemanth Madhavarao, Lei Xu, Weidong Shi  
**Category**: cs.AI  
**Published**: 2026-07-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.14416v1  

#### Abstract
The interconnected nature of global financial systems makes them vulnerable to systemic risks, where the failure of a few institutions can trigger catastrophic cascading defaults. Traditional risk models often fail to capture the complex, non-linear dynamics of these networks. While Graph Neural Net...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《CausalGraphX: A Counterfactual Graph Neural Network Framework for Explainable Systemic Risk Assessment》核心总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

全球金融系统具有高度互联性，单个机构的失败可能通过**cascading defaults**（级联违约）引发系统性风险。传统风险模型（如DebtRank、CoVaR）依赖强线性假设，难以捕捉复杂非线性动态。而当前基于**Graph Neural Networks (GNNs)** 的方法虽具备较强预测能力，但存在两大缺陷：

- **缺乏可解释性**（Black-box）：无法为监管者提供干预依据。
- **相关性而非因果性**：学习到的是**spurious correlations**（虚假关联），而非真正的风险传播机制。

这使得现有模型在压力测试和政策制定中应用受限。

---

### **提出了什么新方法或新思路**

作者提出 **CausalGraphX** —— 一种融合**因果推理**与**反事实解释**的图神经网络框架，用于可解释的系统性风险评估。其核心创新包括：

#### ✅ **双阶段架构设计**
1. **Causal Representation Learner**  
   - 基于 **Graph Attention Network (GAT)** 学习节点表征。
   - 引入**对抗正则化**（adversarial regularization）机制，使学到的嵌入 $Z$ 对**混杂变量**（confounders，如区域、行业）不变，从而增强因果有效性。

2. **Counterfactual Explanation Generator**  
   - 通过优化生成**最小干预方案**，回答“若向某银行注入多少资本，即可避免其违约？”等问题。
   - 支持对节点特征（如资本储备）和边权重（如债务敞口）进行修改。

#### ✅ **因果驱动的可解释性**
- 不仅预测违约，还提供**actionable interventions**（可操作建议），例如：“只需提升 Bank B 的资本15%，即可阻断整个传染链”。

---

### **相比现有方法的优势**

| 维度 | 传统模型 / 标准 GNN | CausalGraphX |
|------|---------------------|------------|
| 预测能力 | 中等（线性假设限制） | ✔️ 更高（非线性建模 + GAT 注意力） |
| 可解释性 | 无或后验解释（post-hoc） | ✔️ 内生因果解释 + 反事实生成 |
| 因果有效性 | 易受混杂因素影响 | ✔️ 抗虚假相关，强调真实传播路径 |
| 干预指导性 | 诊断为主 | ✔️ 提供最小成本干预策略 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

由于真实金融网络数据稀缺且敏感，研究采用**大规模合成金融网络**，确保可控性和可复现性：

- **网络拓扑**：
  - **Erdős–Rényi (ER)**：随机连接结构。
  - **Barabási–Albert (BA)**：无标度网络，模拟现实中的“核心-边缘”结构（hub 节点）。
- **规模**：$N = \{500, 1000, 2000\}$ 家机构。
- **节点特征 $X$**：12维金融指标，包括：
  - Tier 1 capital ratio, leverage ratio, liquidity coverage ratio, NPL ratio, ROA, etc.
- **边权重 $W$**：表示机构间敞口，服从对数正态分布 $\text{LogNormal}(\mu=10.5, \sigma=2)$。
- **标签生成**：使用改进版 **Eisenberg-Noe (E-N) 清算模型**模拟初始冲击（1%机构违约）后的级联过程，得到最终违约状态 $Y \in \{0,1\}$。
- **引入混杂变量 $C$**：人为设定区域标识作为混杂因子，制造虚假相关以测试因果鲁棒性。

---

### **实验设置和评估指标**

#### **评估任务**
- **系统性风险预测**：预测每个机构是否最终违约。

#### **评估指标**

| 类型 | 指标 | 说明 |
|------|------|------|
| **预测性能** | AUC, F1-score, PR-AUC, MCC | 衡量分类准确性，尤其关注不平衡场景 |
| **反事实质量** | Validity (%) | 成功改变预测的比例（目标：Y=0） |
| | Proximity (L2) | $G'$ 与原始图 $G$ 的距离（越小越好） |
| | Sparsity (L0/L1) | 修改的特征/边数量（越少越可解释） |
| | Plausibility (%) | 干预是否符合真实因果机制（合成数据中已知） |
| | Actionability (%) | 是否为现实中可行的干预措施 |

#### **训练细节**
- 模型结构：3层 GAT，每层8个attention heads。
- 对抗判别器：3层 MLP，预测混杂变量 $C$。
- 优化器：Adam；损失函数结合 BCE + adversarial term。
- 反事实生成：使用梯度优化（Algorithm 1），通过 backpropagation 调整输入 $X', W'$。

---

### **基线方法对比**

| 基线模型 | 类型 | 特点 |
|--------|------|------|
| **Logistic Regression (LR)** | 传统统计模型 | 仅用节点特征和简单网络统计量 |
| **DebtRank (DR)** | 系统性风险经典模型 | 基于反馈机制衡量机构重要性 |
| **GCN** | 图神经网络 | 标准图卷积，邻居均等加权 |
| **GraphSAGE** | 归纳式 GNN | 聚合固定采样邻居 |
| **GAT (Standard)** | 注意力 GNN | 与 CausalGraphX 架构相同但无对抗正则化（$\lambda=0$） |

---

## 3. 主要实验结果和性能指标

### **关键性能数据（见 Table 1 & 2）**

#### 📊 在 BA 网络（N=2000）上的表现（最佳值加粗）：

| Model | AUC | F1 | PR-AUC | MCC |
|-------|-----|----|--------|-----|
| LR | 0.719 | 0.648 | – | – |
| DebtRank | 0.769 | 0.701 | – | – |
| GCN | 0.839 | 0.781 | – | – |
| GraphSAGE | 0.848 | 0.792 | – | – |
| GAT (Standard) | 0.873 | 0.818 | 0.428 | 0.637 |
| **CausalGraphX** | **0.908** | **0.859** | **0.516** | **0.701** |

> ✅ **AUC 提升 4.0%**，F1 提升近 5%，表明更强的预测能力。

---

### **与基线方法的对比结果**

- **全面超越所有基线**：在 ER 和 BA 两种拓扑下，CausalGraphX 在 AUC 和 F1 上均取得最优结果。
- 尤其在更大网络（N=2000）上优势更明显，显示良好的**可扩展性**。
- 即使与同架构的 GAT 相比，也因引入**因果正则化**显著提升性能。

---

### **消融实验结果（Ablation Study）**

#### 🔍 对抗正则化参数 $\lambda$ 的影响（Figure 2）
- 当 $\lambda = 0.1$ 时，达到**AUC 与 Causal Plausibility 的最佳平衡**。
- 过大的 $\lambda$ 会损害预测精度，过小则无法有效去除虚假相关。

#### 🧪 鲁棒性测试（Table 5）——在扰动下的 AUC 下降

| 扰动类型 | GAT | CausalGraphX |
|--------|-----|---------------|
| 高斯噪声 ($\sigma=0.1$) | -0.042 | **-0.019** |
| 特征丢弃（10%） | -0.068 | **-0.031** |
| 边丢弃（5%） | -0.091 | **-0.044** |
| 对抗攻击 | -0.124 | **-0.072** |

> ✅ CausalGraphX 显著更鲁棒，验证了**因果表征的稳定性优势**。

---

### **反事实解释质量（Table 3）**

| Metric | GAT | CausalGraphX | Improvement |
|--------|-----|----------------|-------------|
| Validity (%) | 97.3 | **99.1** | +1.8% |
| Proximity (L2) | 0.168 | **0.139** | ↓17.3% |
| Sparsity (L0) | 4.82 | **2.94** | ↓39.0% |
| Plausibility (%) | 89.3 | **91.6** | +26.5% |
| Actionability (%) | 72.4 | **91.6** | +26.5% |

> ✅ 生成的反事实更**接近原图、更稀疏、更合理且更具行动指导意义**。

---

### **实际干预效果示例（Figure 3）**
- 原始情景：Bank S 失败 → 引发连锁反应 → 共4家银行违约。
- CausalGraphX 推荐：仅需将 **Bank B 的资本增加15%**。
- 结果：传染被阻断，仅 Bank S 违约，总损失从 \$142B 降至 \$31B。
- **投入 \$4.2B 资本，减少 \$111B 损失 ⇒ ROI ≈ 33.8x**

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **CausalGraphX 显著优于现有方法**：在预测准确率（AUC/F1）和解释质量（sparsity, plausibility）上均领先。
2. ✅ **对抗正则化有效提升因果有效性**：能抑制混杂变量带来的虚假相关，提高模型鲁棒性。
3. ✅ **反事实解释具备实际政策价值**：生成的干预方案**稀疏、低成本、高回报**，适合监管决策支持。
4. ✅ **因果与可解释性可协同优化**：并非牺牲预测性能换取解释性，而是二者共同提升。

---

### **方法的局限性**

1. **依赖合成数据验证因果性**：虽然在合成环境中验证了 plausibility，但在真实世界中因果机制未知，难以完全验证。
2. **静态图假设**：当前模型处理的是静态网络，未考虑时间演化（如动态借贷关系）。
3. **反事实搜索空间有限**：目前只允许调整资本和敞口，未涵盖更复杂的制度性干预（如流动性支持机制）。
4. **计算开销较高**：反事实生成需多次前向/反向传播（平均 ~18 秒 @ N=2000，见 Table 4），实时性有待提升。

---

### **未来工作方向**

1. **Temporal GNN extension**：引入时序建模能力，构建 **Temporal CausalGraphX**，捕捉动态风险传播。
2. **Multi-layer Financial Networks**：建模多种类型的金融联系（信贷、衍生品、股权等）构成的多层网络。
3. **Real-world Deployment**：与央行或金融监管机构合作，在半真实数据上试点部署。
4. **Interactive Policy Simulator**：开发可视化工具，供监管者交互式探索不同干预策略的效果。

---

> **总结一句话**：  
> CausalGraphX 成功将 **GNN 的表达能力**、**因果推理的稳健性** 与 **反事实解释的可操作性** 融为一体，为下一代可信赖的金融风险管理系统提供了坚实框架。

</details>

---

### 11. [Collaborative Spatial Learning with Multi-LLM Agents in Networked Social Experiments](https://arxiv.org/abs/2607.14574)

**Authors**: Hao He, Chris J. Kuhlman, Xinwei Deng  
**Category**: cs.AI  
**Published**: 2026-07-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.14574v1  

#### Abstract
Collective problem solving often requires that group members consider the tradeoff between exploitation of known solutions and exploration for new ones, where information of known solutions can be disseminated among individual members through communication networks. The Mason--Watts experiment (PNAS...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Collaborative Spatial Learning with Multi-LLM Agents in Networked Social Experiments*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
本论文探讨了**网络结构如何影响多LLM代理（Multi-LLM Agents）在协作空间搜索任务中的集体表现**，特别是是否能复现人类群体中观察到的“网络效率效应”（network-efficiency effect）。该效应指：在较短路径长度（shorter-path）的通信网络中，群体的集体绩效更高。

此前研究已在人类被试中验证这一现象（如Mason-Watts实验），但在**大语言模型（LLM）代理系统中尚未系统测试**。本文填补了这一空白，并进一步比较了LLM代理与机制化模型（如Bayesian Optimization）的行为差异。

---

### 🧩 提出的新方法与新思路

1. **首次将多LLM代理置于可变网络拓扑中进行顺序空间学习实验**  
   使用Mason-Watts实验的8种3-regular图结构（共16个节点），构建了一个受控的协作搜索环境，用于评估不同网络结构对LLM集体性能的影响。

2. **引入“随机初始化指令”（Random Initialization Instruction, RI）作为prompt层面的干预手段**  
   在第一轮选择时添加一句提示：“若为第一轮，请从网格中均匀随机选择一个位置。”以此探究初始化策略对后续集体行为的影响。

3. **设计机制化的Bayesian Optimization代理作为强基线**  
   构建基于高斯过程（Gaussian Process）的EI（Expected Improvement）和UCB（Upper Confidence Bound）代理，接收与LLM相同的输入信息（自身及邻居历史），实现探索-利用权衡的显式建模。

4. **构造60个可控复杂度的二维fitness landscape数据集**  
   包括低、中、高三类复杂度（通过局部峰值数量K和尾部分布控制），确保实验结果不受landscape随机性干扰。

---

### 🔍 相比现有方法的优势

| 方面 | 优势 |
|------|------|
| **实验设计** | 首次系统性地在固定网络拓扑下测试LLM代理的空间协作能力，具备与人类实验直接可比性 |
| **可重复性** | 所有landscape生成具有确定性种子；使用开源权重模型gpt-oss-120b，提升复现性 |
| **分析深度** | 不仅比较最终收益，还深入分析copying行为、空间多样性、探索-利用动态等机制性因素 |
| **基准全面** | 同时包含随机代理、人类数据、机制化BO代理，形成多层次对比体系 |

---

## 2. 核心实验方法和设置

### 📚 数据集与任务设置

- **任务名称**：Wildcat Wells 游戏（源自Mason & Watts, 2012）
- **搜索空间**：离散二维网格 $\{0,\dots,100\}^2$，共 $101 \times 101$ 单元格
- **目标函数**：隐藏的fitness landscape $f: \Omega \to [0,100]$，全局最大值为100且唯一
- **游戏轮数**：$T = 15$ 轮同步决策
- **代理数量**：每组 $N = 16$ 个代理，构成固定3-regular图（每个节点有3个邻居）

#### Landscape Construction（三阶段生成）
| 复杂度 | 局部峰值数 $K$ | >80的单元比例 | ≤20的单元比例 |
|--------|----------------|---------------|----------------|
| Low    | 0              | ~20%          | ~30%           |
| Moderate | 3            | ~10%          | ~40%           |
| High   | 8              | ~5%           | ~50%           |

> 共生成60个landscape（每类复杂度20个），由Perlin噪声+主导峰+次级峰组合而成，保证可比性和难度梯度。

---

### 🧪 实验设置与评估指标

#### 代理类型（共5类 + 人类参考）

| 代理类型 | 描述 |
|---------|------|
| **LLM (default)** | gpt-oss-120b，默认prompt，无特殊初始化指导 |
| **LLM-RI** | 同上，但在第1轮增加“随机初始化”指令 |
| **EI / UCB** | 基于GP的Bayesian Optimization代理，使用Expected Improvement或UCB采集函数 |
| **Random** | 完全随机选择（忽略所有社会信息） |
| **Human** | 来自原始Mason-Watts实验的29组人类数据（共232场游戏） |

#### 网络结构
- 使用原始实验中的**8种3-regular图**（A–H），按平均路径长度（APL）排序（2.20 → 3.87）
- APL越小表示网络效率越高（信息传播更快）

#### 主要评估指标
| 指标 | 定义 | 用途 |
|------|------|------|
| **Cumulative Game-Level Mean Payoff ($\bar{f}_{cum}$)** | $\frac{1}{NT}\sum_{i=1}^{N}\sum_{t=1}^{T} f(x_i^{(t)})$ | 衡量整体绩效，匹配激励结构 |
| **Final-Round Group-Mean Payoff ($\bar{f}_T$)** | 第15轮平均收益 | 反映收敛质量 |
| **Near-Peak Discovery Rate** | 团队运行最大值达到≥99的比例 | 衡量是否找到接近最优解 |
| **Copying Rate** | 当前轮猜测落在任一邻居上一轮附近（距离≤3）的比例 | 分析社会学习行为 |
| **Spatial Diversity ($D(t)$)** | 轮次$t$中所有代理猜测之间的平均欧氏距离 | 度量探索广度 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

| 代理类型 | 平均累计收益 ($\bar{f}_{cum}$) | 近峰发现率（≥99） | Copying Rate | Spatial Diversity（终轮） |
|--------|-------------------------------|--------------------|--------------|----------------------------|
| **UCB** | ~97–100 | 91–99% | 74% | ~2–3 |
| **EI** | ~97–100 | 91–99% | 75% | ~2–3 |
| **LLM-RI** | ~93–97 | 62–76% | 86% | ~2–3 |
| **LLM (default)** | ~84–91 | 25–54% | 77% | ~2–3 |
| **Random** | ~39–47 | <2% | <1% | ~52（恒定） |
| **Human** | —（参考方向一致） | — | 54% | ~10 |

> 注：具体数值随landscape复杂度变化，此处为趋势范围。

---

### ⚖️ 与基线方法的对比结果

#### （1）**网络效率效应的存在性**

| 代理类型 | 是否存在显著网络效应？（APL↓ ⇒ 收益↑） |
|--------|------------------------------------------|
| **LLM (default)** | ❌ 不显著（$p=0.619$） |
| **LLM-RI** | ✅ 显著（$\beta = -1.37, p=0.003$） |
| **EI** | ✅ 显著（$\beta = -1.26, p=0.005$） |
| **UCB** | ✅ 显著（$\beta = -0.99, p=0.026$） |
| **Human** | ✅ 显著（$\beta = -4.65, p=0.010$） |

> 结论：只有当LLM被明确指示**随机初始化**时，才表现出类似人类的网络效率效应。

#### （2）**性能排名**
$$
\text{EI ≈ UCB > LLM-RI > LLM (default) >> Random}
$$
- BO代理显著优于所有LLM配置
- 单句prompt修改（RI）使LLM提升约 **9.44单位累计收益**
- 此增益是**整个网络拓扑效应范围的3倍以上**

#### （3）**消融实验：初始化的影响**
- 默认LLM在99.96%的情况下第一轮选择中心点 $(50,50)$
- 导致初始空间覆盖极差，浪费早期社会信息价值
- 添加“随机初始化”指令后：
  - 初始分散性提高
  - 更早利用邻居信息
  - 显著恢复网络结构对绩效的影响

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **LLM代理能否展现网络效率效应取决于prompt设计**  
   - 默认LLM因集中启动而无法体现网络优势
   - 加入**单句随机初始化指令**即可恢复显著的网络效应

2. **prompt工程的影响远超网络结构本身**  
   - 初始化带来的性能提升 > 所有8种网络之间差异的3倍
   - 表明在多代理系统中，**个体行为先验比通信结构更重要**

3. **LLM的社会学习行为偏向“位置模仿”而非“奖励驱动”**  
   - Copying rate高达77–86%，但与payoff相关性弱（$r=0.25$）
   - 相比之下，BO代理的copying与其收益高度正相关（$r≈0.87–0.90$）
   - 说明LLM更多是在复制坐标，而不是学习“哪些动作带来高回报”

4. **空间多样性迅速崩溃，削弱后期网络作用**  
   - 所有智能代理在几轮内即收敛至同一区域
   - 导致后期neighbor信息冗余，网络结构不再重要
   - 最终轮次的网络效应消失（所有$p > 0.4$）

5. **Bayesian Optimization代理全面超越LLM**  
   - 在相同信息条件下，机制化探索-利用策略更高效
   - 表明当前LLM在结构化搜索任务中仍落后于专用算法

---

### ⚠️ 方法的局限性

| 局限性 | 说明 |
|-------|------|
| **单一LLM模型与prompt模板** | 使用gpt-oss-120b和固定prompt，结果可能不具备泛化性 |
| **非公开的人类lanscape不可比** | 人类使用的landscape未公开，绝对收益无法直接对比 |
| **同步更新机制限制动态性** | 异步交互可能改变收敛速度和网络效应强度 |
| **仅限二维空间任务** | 无法推广到代码生成、规划等更复杂的协作场景 |
| **短期回合限制（15轮）** | 长期协作动态未被充分考察 |

---

### 🔮 未来工作方向

1. **扩展至其他LLM架构与prompt策略**  
   测试不同模型（如Llama系列）、思维链（Chain-of-Thought）、反思机制对协作行为的影响。

2. **引入动态网络结构或自组织连接**  
   探索LLM是否能在运行时主动调整通信对象（endogenous network formation）。

3. **设计异步或多阶段协作协议**  
   模拟真实团队协作节奏，研究时间延迟对信息扩散的影响。

4. **应用于更丰富任务**  
   如协同编程（ChatDev）、科学假设生成、战略博弈等，检验空间学习之外的协作模式。

5. **结合认知建模解释LLM行为**  
   将LLM视为“具身化认知体”，用计算心理学工具分析其信念更新与社会推理机制。

---

> 💬 **一句话总结**：  
> 本文揭示了**LLM代理的集体表现极度依赖prompt引导**，即使简单的“第一轮随机选择”指令也能释放网络结构的潜力；然而，在结构化空间搜索任务中，**当前LLM仍远不如机制化Bayesian Optimization方法高效**，其社会学习行为更像盲目模仿而非理性决策。

</details>

---

### 12. [SmartRAG: Native Graph-Based RAG for Mobile Device](https://arxiv.org/abs/2607.14661)

**Authors**: Zhihan Jiang, Meng Li, Shenghao Liu, Keran Li, Ruiben Zhou, Xianjun Deng, Shuai Wang, Haipeng Dai  
**Category**: cs.AI  
**Published**: 2026-07-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.14661v1  

#### Abstract
Deploying large language models (LLMs) as personal assistants on mobile devices demands privacy, low latency, and offline availability, yet the computational cost of giant models clashes with strict edge-hardware budgets. We argue that this tension cannot be resolved by model compression alone; it r...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*SmartRAG: Native Graph-Based RAG for Mobile Device*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在移动端部署 **Large Language Models (LLMs)** 作为个人助手面临三大挑战：**隐私保护、低延迟响应、离线可用性**。然而，大型模型的计算开销与移动设备的硬件资源（内存、功耗、算力）严重不匹配。

现有基于 **Retrieval-Augmented Generation (RAG)** 的方法大多依赖于简单的向量检索（naive vector retrieval），难以支持需要跨段落证据聚合、实体关联和多跳推理（multi-hop reasoning）的复杂查询。而云上先进的图结构 RAG 方法又严重依赖数十亿参数的大模型进行图构建与推理，在边缘设备上不可行。

因此，核心问题是：**如何在资源受限的移动设备上实现高效、可扩展且支持复杂推理的结构化 RAG？**

---

### 提出了什么新方法或新思路
作者提出 **SmartRAG** —— 一种完全运行在设备端的、基于图结构的 RAG 框架，其核心思想是将智能分解为四个协同模块：

- **Perception（感知）**：轻量级模块负责从文本中提取结构化知识（实体与关系）。
- **Memory（记忆）**：构建并维护一个三层的、保留来源信息的 **MRGraph** 知识图谱。
- **Focus（聚焦）**：通过混合检索策略（图遍历 + 词法匹配 + 密集语义搜索）定位相关证据。
- **Thinking（思考）**：仅在必要时调用 LLM 执行高价值语义任务（如标签生成、计划制定、答案合成）。

#### 关键技术创新：
- **EvoNER**：一个持续可学习的命名实体识别器（continually learnable NER），通过教师蒸馏（teacher-distilled updates）增量更新标签体系，无需重新训练主干 LLM 即可识别新出现的实体类型。
- **MRGraph**：三层次知识图结构，支持持续写入与高效检索，并显式保留原始文本来源（provenance-preserving）。
- **Hybrid Retrieval Pipeline**：结合图遍历、词法匹配和密集检索的混合检索机制，提升多跳问答能力。
- **功能解耦设计**：将知识抽取、组织、检索与生成分离，使系统能在轻量模型下协作完成复杂任务。

---

### 相比现有方法的优势
| 维度 | 传统方法 | SmartRAG |
|------|--------|---------|
| **部署可行性** | 多数图 RAG 需要大模型，无法在手机运行 | 完全在普通智能手机上运行（如 OnePlus 13/15） |
| **知识更新能力** | 固定标签体系，难适应新实体 | EvoNER 支持持续学习，动态扩展实体类别 |
| **推理能力** | 向量检索难以支持多跳推理 | 图结构 + 规划机制显著提升 multi-hop QA 性能 |
| **效率控制** | LLM 被频繁调用，成本高 | LLM 仅用于高价值操作，推理成本可控 |

---

## 2. 核心实验方法和设置

### 使用的数据集
在四个标准 QA 基准上进行评估：
- **TriviaQA**：开放域事实型问答
- **Natural Questions (NQ)**：真实用户提问，需精准回答
- **HotpotQA**：多跳推理，需整合多个文档信息
- **MultiHopQA**：专门设计用于评估多步推理能力

---

### 实验设置和评估指标

#### 模型配置
- 主干 LLM 使用两种量化后的轻量模型：
  - **Qwen3-1.7B**（Q6_K 量化）
  - **Ministral3-3B**（Q6_K 量化）
- 所有流程（记忆摄入、索引、检索、生成）均在设备端执行。
- 实验平台：OnePlus 13 和 OnePlus 15 智能手机（Snapdragon 8 Elite 系列芯片，16GB RAM）。

#### 评估指标
- **Correctness (Cr, %)**：使用 LLM-as-judger 协议判断预测是否事实正确（judge model: Claude Haiku 4.5）。
- **Token-F1 (%)**：预测与参考答案之间的 token 级精确率-召回率 F1 分数。

#### 对比基线
分为两类：

##### （1）LLM-only 方法（无检索）
- Qwen3-1.7B / 4B / 32B
- Ministral3-3B / 14B

##### （2）RAG 方法（使用轻量模型）
- **Naive RAG**：标准 dense retrieval
- **InstructRAG**：通过自生成 rationale 进行去噪
- **StableRAG**：缓解检索顺序导致的幻觉
- **TruthfulRAG**：基于 KG 的冲突检测与过滤

> 注：所有 RAG 基线均在同一量化后端（llama.cpp）下模拟运行，以保证公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| 方法 | TriviaQA (Cr) | NQ (Cr) | HotpotQA (Cr) | MultiHopQA (Cr) |
|------|---------------|--------|----------------|--------------------|
| Qwen3-32B (LLM-only) | 51.45% | 40.15% | 33.25% | 31.54% |
| Ministral3-14B (LLM-only) | 60.70% | 37.80% | 22.40% | 20.12% |
| **SmartRAG (Qwen3-1.7B)** | **55.00%** | **66.68%** | **63.93%** | **50.17%** |
| **SmartRAG (Ministral3-3B)** | **59.74%** | **63.86%** | **66.74%** | **57.60%** |

> ✅ **关键发现**：尽管 SmartRAG 使用的是仅 **1.7B 或 3B 参数**的量化模型，其在 NQ、HotpotQA 和 MultiHopQA 上的表现**超过甚至远超 32B 级别的纯大模型**。

---

### 与基线方法的对比结果
- 在 **NQ、HotpotQA、MultiHopQA** 上，SmartRAG 显著优于所有基线 RAG 方法（包括 InstructRAG、TruthfulRAG 等）。
- 在 **TriviaQA** 上表现良好但优势不如其他任务明显，说明其强项在于**知识密集型、需组合分散证据的任务**。
- 特别是在 **MultiHopQA** 上，SmartRAG 达到约 **50–57% Cr**，而其他 RAG 方法普遍低于 45%，显示出其对多跳推理的强大支持。

---

### 消融实验结果（Ablation Study）
在 MultiHopQA 上对三个核心组件进行消融：

| 变体 | Token-F1 |
|------|----------|
| Full SmartRAG | **0.5484** |
| - MRGraph（移除结构化记忆） | 0.4151 ❌（下降 24.3%） |
| - THINKING_PLAN（移除 LLM 规划） | 0.5069 ❌ |
| - Hybrid Retrieval（仅图检索） | 0.4749 ❌ |

> 🔍 **结论**：
> - **MRGraph 是最重要的组件**，移除后性能大幅下降。
> - **多跳规划（planning）有助于更精准地获取证据**。
> - **混合检索优于单一图检索**，表明多种检索通道互补有效。

---

## 4. 关键结论和发现

### 主要发现
1. **结构化记忆比单纯增加参数更重要**：在设备受限条件下，优化外部记忆结构（如 MRGraph）可以显著提升小模型在复杂 QA 任务上的表现，甚至超越大模型。
2. **模块化解耦是可行路径**：将知识提取、组织、检索与生成分离，允许轻量模块协同工作，避免对单一强大 LLM 的依赖。
3. **持续学习能力至关重要**：EvoNER 实现了在不重训主干 LLM 的前提下动态吸收新实体，增强了系统的长期适应性。
4. **混合检索优于纯语义或纯图检索**：结合图结构、词法与向量信号，可在精度与覆盖率之间取得更好平衡。

---

### 方法的局限性
1. **当前规划器为启发式规则**，不能均匀适用于所有问题类型；未来可引入学习式规划策略。
2. **端到端延迟仍较高**：尤其是 retrieval-conditioned prefilling 阶段主导了 TTFT（平均首次 token 时间达 ~37 秒），影响交互体验。
3. **某些领域实体识别仍弱**：如 “AI” 和 “Politics” 类型的 NER 表现不佳，需改进聚类与类型一致性。
4. **仅限英文评测**：尚未验证多语言场景下的有效性。

---

### 未来工作方向
- 设计更智能的动态检索策略，平衡图深度与检索精度。
- 引入证据压缩技术（evidence compression）降低 prefilling 成本。
- 探索端到端训练的轻量规划器（learned planner）。
- 扩展至多语言和个人化知识图谱构建。
- 加强设备安全机制（如加密存储），应对本地知识图谱泄露风险。

---

> 💡 **总体评价**：  
> SmartRAG 提供了一条切实可行的路径，使得高性能、隐私友好、支持复杂推理的个人 AI 助手能够在普通智能手机上**完全离线运行**。它标志着从“依赖更大模型”向“更优系统架构”的范式转变，是迈向真正个性化、可持续演化的 on-device AI 的重要一步。

</details>

---

### 13. [Reachability-Aware Pretraining for Efficient Target-Oriented Path Exploration in Temporal Knowledge Graph Reasoning](https://arxiv.org/abs/2607.14886)

**Authors**: Chien-Liang Liu, Tsao-Lun Chen  
**Category**: cs.AI  
**Published**: 2026-07-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.14886v1  

#### Abstract
Temporal Knowledge Graph (TKG) reasoning under the extrapolation setting focuses on forecasting future time-stamped events (facts) from historical data in a temporal knowledge graph. Existing approaches, reinforcement learning (RL)-based multi-hop reasoning methods are prominent for TKG reasoning be...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Reachability-Aware Pretraining for Efficient Target-Oriented Path Exploration in Temporal Knowledge Graph Reasoning*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **Temporal Knowledge Graph (TKG)** 推理任务中，尤其是在**外推（extrapolation）设定**下，现有的基于 **Reinforcement Learning (RL)** 的多跳推理方法面临两个核心挑战：
- **奖励稀疏（sparse rewards）**：大多数采样的路径无法到达目标实体，导致终端奖励极少。
- **探索效率低下（inefficient exploration）**：由于时间演化图结构庞大且动作空间随时间变化（temporal validity 约束），搜索空间呈指数增长，大量计算资源被浪费在不可达路径上。

这些问题严重阻碍了 RL 的训练效率和最终性能。

---

### 🚀 提出的新方法：RAPTOR
作者提出 **RAPTOR (Reachability-Aware Pretraining for Efficient Target-ORiented Path Exploration)** ——一种**自监督预训练框架**，用于提升 RL-based 多跳 TKG 推理的效率与效果。

#### 核心思想
通过引入 **“可达性感知”（reachability-aware）的归纳偏置（inductive bias）**，在正式进行 RL 微调前，预先教会 agent 判断哪些候选动作（即 outgoing edges）**有可能在有限步数内、满足时间约束地到达目标实体**。

#### 方法流程
1. **可达实体标注算法（Reachable Entity Labeling）**  
   从目标实体 $ o_q $ 出发，反向执行带时间约束的 BFS，记录每个实体在特定 hop 距离内的最晚可达时间戳。
2. **自监督预训练（Self-Supervised Pretraining）**  
   使用上述标签构建二分类任务：预测当前动作是否属于“可达路径”的一部分，使用 Binary Cross-Entropy (BCE) 损失优化策略网络。
3. **下游 RL 微调**  
   将预训练后的模型作为初始化，继续用 actor-critic RL 算法进行微调。

---

### 🔍 相比现有方法的优势
| 对比维度 | RAPTOR vs. 现有方法 |
|--------|------------------|
| **适用场景** | 首个专为 **TKG 上 RL-based 多跳推理**设计的预训练方法；而如 DeepPath、SSRL 等仅适用于静态 KG，未考虑时间约束 |
| **引导方式** | 在 RL 前提供强先验，避免边训练边学习探索策略；相比 TITer、TPath 等在 RL 中加入 reward shaping 或示范路径的方法，干扰更小 |
| **效率提升** | 显著减少无效探索，加速收敛，提高样本利用率 |
| **通用性** | 可作为插件式模块集成到多种 RL-based 多跳推理架构中 |

---

## 2. 核心实验方法和设置

### 📊 数据集
在三个标准 TKG 外推基准上进行评估：
- **ICEWS14**
- **ICEWS05-15**
- **ICEWS18**

所有数据均来自 ICEWS 事件数据库，按时间划分 train/valid/test，并保留时间粒度为 **1 day**。

> 数据统计详见附录 Table 2：
> - 实体数从 7k 到 23k 不等
> - 时间跨度不同，规模递增

---

### 🧪 实验设置
- **最大路径长度（hop budget）**：$ K = 3 $
- **嵌入维度**：实体 128，关系（含时间编码）128 + 48，step embedding 32
- **预训练阶段**：40 epochs 自监督 BCE 训练
- **RL 微调阶段**：400 epochs，使用 actor-critic + entropy regularization
- **优化器**：Adam ($ \text{lr} = 1e^{-3} $)，batch size = 512
- **动作剪枝**：每步保留 timestamp 最大的 top-150 候选动作

---

### 📏 评估指标
采用 **time-aware filtered setting** 下的标准指标：
- **MRR (Mean Reciprocal Rank)**
- **Hits@1, Hits@3, Hits@10**

该设置只过滤与查询相同时间戳的历史事实，比 raw 或普通 filtered 更合理。

---

### ⚔️ 基线方法对比
分为两类：

#### ❖ 非多跳推理基线（Non-multi-hop）
- xERTE
- TLogic
- CyGNet
- RE-NET
- RE-GCN

#### ❖ 多跳推理基线（Multi-hop + RL）
- TAgent
- TITer（含时间奖励塑形）
- TPath（加路径奖励）
- TITer-150（作者实现，限制动作大小）
- **Pure-RL**：本文模型但无 RAPTOR 预训练（消融对照）

> 多数结果复现自已有文献（Li et al., 2022; Zheng et al., 2023），确保公平比较。

---

## 3. 主要实验结果和性能指标

### 📈 性能汇总（见 Table 1）

| Model | ICEWS14 MRR | ICEWS05-15 MRR | ICEWS18 MRR |
|-------|-------------|----------------|-------------|
| RE-GCN (best baseline) | 40.39 | 48.03 | **30.58** |
| Pure-RL (our backbone) | 42.41 | 47.74 | 29.87 |
| **RL-RAPTOR (ours)** | **42.86*** | **48.04*** | **30.25*** |

> *表示相对于 Pure-RL 达到统计显著性（p < 0.05）

#### 关键观察：
- 在 **ICEWS14 和 ICEWS05-15** 上，**RL-RAPTOR 全面超越所有基线**，取得最佳 MRR 和多数 Hits@k 指标。
- 在 **ICEWS18** 上略低于 RE-GCN（非多跳模型），但仍优于其他多跳方法，且在 Hits@1/3 上表现更强。
- 即使不是 Hits@10 最高，整体表现**稳定且持续领先**。

---

### 🔬 消融实验（Ablation Study）
在 ICEWS14 上研究不同预训练 epoch 数的影响（10–50 epochs）：

#### 发现：
- **预训练越长，RL 阶段起始奖励越高** → 更快进入高效探索阶段
- 仅 **10 轮预训练**即可让模型在 60 epoch 内达到 Pure-RL 超过 100 epoch 才能达到的平均奖励水平
- 验证集 MRR 在早期明显更高，说明 RAPTOR 提供了**更好的参数初始化**

> 结论：RAPTOR 显著提升了 **sample efficiency** 和 **training stability**

---

### ⏱️ 训练效率分析（Training Efficiency）
- **收敛速度更快**：RL-RAPTOR 在更少训练轮次内达到甚至超过基线的最终性能
- **奖励曲线更平滑上升**，表明探索更有方向性
- 在较难的数据集（如 ICEWS18）上优势更明显，尤其体现在 MRR 提升上

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **可达性预训练有效缓解了 RL 中的稀疏奖励问题**  
   通过提前识别并抑制不可达动作，大幅减少了无效探索。
   
2. **RAPTOR 是首个面向 TKG 多跳推理的自监督预训练方法**  
   引入了符合任务特性的归纳偏置（temporal + hop-constrained reachability），具有较强可解释性和泛化潜力。

3. **显著提升训练效率与最终性能**  
   在多个标准数据集上一致优于主流基线，尤其在 MRR 和低阶 Hits 上优势明显。

4. **良好的兼容性与即插即用特性**  
   可无缝集成进现有 RL-based 多跳推理框架，无需复杂结构调整。

---

### ⚠️ 局限性（Limitations）
1. **目前仅适用于 RL-based 多跳推理框架**，尚未验证其对其他范式（如 GNN-based 或 sequence-to-sequence）的有效性。
2. **预处理开销较大**：可达实体标注需对每个训练样本运行反向 BFS，可能成为大规模 TKG 上的瓶颈。
3. **模型主干相对简单**：出于公平比较目的，采用了 LSTM + MLP 架构，未结合更先进的模型结构（如 Transformer 或 memory network）。

---

### 🔮 未来工作方向
1. **优化可达性标注阶段的效率**  
   如设计近似算法或增量更新机制以适应动态或超大规模 TKG。
   
2. **扩展至更强的 backbone 模型**  
   将 RAPTOR 应用于基于 Transformer 或图注意力网络的高级推理模型中。

3. **探索更多类型的自监督信号**  
   如路径语义一致性、时序模式匹配等，进一步丰富预训练目标。

4. **应用于真实世界场景**  
   如金融风险传播预测、疾病传染链追踪等需要可解释路径支持的任务。

---

## ✅ 总结
**RAPTOR** 成功将“可达性判断”这一人类常识引入 TKG 多跳推理系统，通过一个轻量级但高效的自监督预训练阶段，显著提升了 RL 智能体的学习效率和推理能力。它不仅是一个实用的技术改进，也为未来构建**高效、可解释、低样本依赖**的时间图推理系统提供了新思路。

</details>

---

### 14. [Long-Context Fine-Tuning with Limited VRAM](https://arxiv.org/abs/2607.15105)

**Authors**: Vladimir Fedosov, Aleksandr Sazhin, Artemiy Grinenko, Frank Woernle  
**Category**: cs.AI  
**Published**: 2026-07-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.15105v1  

#### Abstract
Parameter-efficient fine-tuning reduces model and optimizer memory, but dense attention still makes long training sequences expensive. We combine Hierarchical Global Attention (HGA) with segment-wise backpropagation and tiered KV storage. Only the active segment remains differentiable in VRAM; older...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Long-Context Fine-Tuning with Limited VRAM*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 **long-context fine-tuning** 受限于 **VRAM 容量**，尤其是 **dense attention** 在长序列训练中导致显存占用急剧上升。尽管 QLoRA 等参数高效微调方法减少了模型和优化器内存，但 **attention 和 backpropagation 所需的中间状态** 仍随上下文长度平方增长，成为瓶颈。

本文旨在解决：**如何在有限 VRAM 下实现超长上下文（>16K tokens）的 fine-tuning**。

---

### 🚀 提出的新方法与创新思路

作者提出一种结合以下三项技术的综合方案：

1. **Hierarchical Global Attention (HGA)**  
   - 替换原始 Transformer 的 dense attention。
   - 将上下文划分为固定大小的 **chunks**（如 64 tokens）和更小的 **groups**（如 8 tokens）。
   - 构建两级 key summaries（chunk-level 和 group-level），通过 **content-based routing** 选择最相关的 chunks 和 groups。
   - 最终只加载被选中的 **exact token K/V** 进行 attention 计算，而非整个历史。

2. **Segment-wise Backpropagation（Truncated BPTT）**  
   - 将长序列切分为多个 segment（如 2K tokens）。
   - 每个 segment 单独前向传播、计算损失并反向传播。
   - segment 边界处的历史 K/V 被 detach，不保留梯度，从而限制 VRAM 中的可微分状态。

3. **Tiered KV Storage（分层 KV 存储）**  
   - 当前活跃 segment 的 K/V 保留在 VRAM 并参与梯度计算。
   - 更早的历史 K/V 被 detach 后移至 **RAM 或 NVMe**。
   - HGA 路由时按需从外部存储加载 selected chunks/groups 的 K/V。

> 💡 **核心思想**：将“完整历史”与“GPU 工作集”解耦——**只有当前 segment 和少量路由选中的历史 token 驻留 VRAM**。

---

### 🔍 相比现有方法的优势

| 方法 | 局限性 | HGA 的优势 |
|------|--------|-----------|
| **Dense Attention** | 显存消耗 $O(L^2)$，无法扩展到长上下文 | 显存仅随 segment 长度增长，历史容量由 RAM/NVMe 决定 |
| **Gradient Checkpointing / FlashAttention** | 减少激活内存，但仍处理全部历史交互 | 显著减少 query-key interaction 数量 |
| **LongLoRA** | 使用 shift window 等稀疏模式，可能丢失全局依赖 | HGA 是 content-aware 的 exact-token 路由，保留关键历史信息 |
| **QLoRA alone** | 仍受限于 dense attention 的序列状态 | 结合 HGA 实现真正意义上的 long-context 微调 |

> ✅ **优势总结**：
> - 支持 **远超 VRAM 容量的训练上下文**（实测达 16K，评估达 131K）
> - 在相同训练长度下，**学习质量与 dense training 几乎无差异**
> - 随着上下文增长，**吞吐有望超越 dense attention**

---

## 2. 核心实验方法和设置

### 📚 数据集
- **PG19**：学术书籍语料，用于 long-context language modeling。
  - 使用官方 train/validation split。
  - 文档普遍较长，适合测试长上下文能力。

---

### ⚙️ 实验设置

| 项目 | 设置 |
|------|------|
| **Model** | Qwen3-8B（4-bit NF4 QLoRA） |
| **LoRA Targets** | Q, K, V, O, gate, up, down projections |
| **Hardware** | Quadro RTX 5000（16GB VRAM） |
| **Training Context Length** | 最高 16,384 tokens（dense 方法在 4K 失败） |
| **Segment Length** | 2,048 tokens（TBPTT 切片） |
| **Chunk Size** | 64 tokens；Group Size：8 tokens |
| **Routing Budget** | 默认 top-8 chunks 被选中 |
| **Precision & Optimizer** | FP16 / PagedAdamW8bit |
| **Environment** | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

---

### 📊 评估指标

| 指标 | 描述 |
|------|------|
| **Loss (nat)** / **Perplexity (PPL)** | 主要语言建模质量指标，使用 dense attention 作为 readout（确保公平比较） |
| **Throughput (tokens/s)** | 训练效率 |
| **Peak VRAM Usage (GB)** | 显存开销 |
| **Routing Cost** | HGA eval 与 dense eval 的 loss 差值，衡量路由带来的精度损失 |
| **Selection Density** | 被选中参与 attention 的历史 token 比例 |
| **KV Saving** | 减少的 K/V 存储交互比例 |
| **Retrieval Recall (%)** | 使用 RULER-style suite 测试 passkey/multikey/multivalue 任务 |

---

### 🔁 基线方法对比

- **Dense-trained adapter**：使用原生 dense attention 微调（仅能在 ≤2K 上运行）
- **HGA-trained adapter**：使用 HGA + TBPTT + tiered KV 微调
- **Stock model**：未微调的原始 Qwen3-8B 模型

> 所有 adapter 在 **相同数据顺序、随机种子、超参** 下训练，保证可比性。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ▶ 表 1：不同上下文长度下的可行性（16GB GPU）

| Sequence | HGA Peak VRAM | Dense Status | HGA Status |
|----------|----------------|---------------|------------|
| 2,048    | 11.76 GB       | OK            | OK         |
| 4,096    | 14.69 GB       | OOM           | OK         |
| 8,192    | 15.26 GB       | OOM           | OK         |
| 16,384   | 15.28 GB       | OOM           | OK         |

> ✅ **结论**：HGA 成功将最大可训练上下文从 **2K 提升至 16K**，而 dense 方法在 4K 即 OOM。

---

#### ▶ 表 2：短上下文下的训练效率（2K）

| Mode | Throughput (tok/s) | Peak VRAM (GB) |
|------|--------------------|----------------|
| Dense | 207.02             | 11.56          |
| HGA   | 217.75             | 11.76          |

> ✅ **结论**：在 2K 长度下，HGA 不仅可行，且已**略微快于 dense 方法**（+5.2%），表明其效率拐点已到来。

---

#### ▶ 表 3：2K 上训练的质量对比（dense readout）

| Training Method | Loss (nat) | PPL     |
|------------------|-----------|---------|
| HGA-trained      | 2.7405    | 15.495  |
| Dense-trained    | 2.7383    | 15.461  |
| Stock model      | 2.9541    | 19.185  |

> ✅ **结论**：
> - HGA 与 dense 训练的 adapter 在 dense readout 下性能几乎一致（差值 <0.0022 nat）。
> - 两者均显著优于原始模型（PPL ↓19.4%）。

---

#### ▶ 表 4 & 5：HGA evaluation 随上下文增长的表现

| Context | Selection Density | KV Saving | HGA Eval Loss |
|--------|--------------------|-----------|--------------|
| 4K     | 33.1%              | 66.9%     | ~2.71        |
| 8K     | 17.3%              | 82.7%     | ~2.43        |
| 16K    | 8.8%               | 91.2%     | ~2.64        |
| 32K    | 4.4%               | 95.6%     | ~2.74        |

> ✅ **结论**：
> - 随着上下文增长，**selection density 持续下降**，说明 HGA 能有效聚焦关键信息。
> - 尽管 routing cost 上升，但模型在长达 32K 的 evaluation 中仍保持可用。

---

#### ▶ 表 7：路由预算消融实验（4K, HGA-trained）

| Top-k Chunks | Loss | Density | KV Saving | Cache Hit Rate |
|-------------|------|--------|-----------|----------------|
| 2           | 2.7530 | 18.8% | 81.2%     | 87.7%          |
| 8 (default) | 2.7180 | 33.1% | 66.9%     | 93.8%          |
| 32          | 2.7108 | 32.6% | 67.4%     | 95.3%          |

> 🔍 **发现**：
> - quality 在 k=8 时接近饱和，进一步增加收益有限。
> - cache hit rate >87%，验证了 summary caching 设计的有效性。

---

#### ▶ 表 8：Retrieval 性能（dense readout）

| Task       | HGA-trained | Dense-trained |
|-----------|-------------|----------------|
| Passkey   | 100.0%      | 100.0%         |
| Multikey  | 100.0%      | 100.0%         |
| Multivalue| 93.8–100.0% | 93.8–100.0%    |

> ✅ **结论**：两种训练方式在 retrieval 能力上表现相当，**HGA 训练未损害 dense attention 下的信息提取能力**。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **HGA + TBPTT + Tiered KV 可突破 VRAM 限制**  
   - 在 16GB GPU 上成功训练 **16K 上下文**，远超 dense 方法的 2K 极限。
   - 实际上限由 RAM/NVMe 容量决定，非 VRAM。

2. **学习质量与 dense training 几乎等价**  
   - 在 2K 和 4K 上，HGA-trained 与 dense-trained adapter 在 dense readout 下性能差异极小（<0.04 PPL）。
   - 表明 HGA 是一个有效的 **training-time 替代方案**。

3. **效率拐点已出现，长期更具优势**  
   - 在 2K 时 HGA 已略快于 dense。
   - 因 HGA 每 token 的计算量近似恒定，而 dense 为 $O(L)$，**预期在更长上下文中 HGA 将显著领先**。

4. **支持灵活推理模式**  
   - 微调后的 adapter 可选择使用 **dense attention**（兼容现有框架）或 **HGA**（节省推理资源）。

---

### ⚠️ 方法的局限性

1. **Long-horizon causal leakage**  
   - 当前 HGA 实现在超长训练（>100M tokens）后可能出现 **causal leakage**：chunk 内多个 query 共享 routing 决策，导致早期 token 间接获取未来信息。
   - 目前适用于 fine-tuning，**不推荐用于从零预训练**。

2. **实验范围有限**  
   - 仅在 Qwen3-8B 和单一 GPU 上验证。
   - dense baseline 缺失于 8K+，无法直接比较质量。
   - NVMe tier 未进行性能 benchmark。
   - routed inference 仍为研究原型，非生产级实现。

---

### 🔮 未来工作方向

1. 设计 **strictly causal routing variant** 以消除 leakage，支持 pretraining。
2. 开发 **production-grade HGA serving engine**，集成 vLLM、SGLang 等主流推理框架。
3. 探索 **adaptive routing budgets** 和 **multi-modal extensions**。
4. 在更多模型（如 Llama、Mixtral）和任务（如 summarization, QA）上验证泛化性。

---

## ✅ 总结

该论文提出了一个实用且高效的 long-context fine-tuning 框架，通过 **HGA + segment-wise BPTT + tiered KV storage** 的组合，在有限 VRAM 下实现了 **16K 上下文训练**，并在质量和效率上媲美甚至超越 dense 方法。它为大模型在真实长文档场景下的适应提供了可行路径，是迈向 **practical ultra-long context learning** 的重要一步。

</details>

---

### 15. [Can We Trust Item Response Theory for AI Evaluation?](https://arxiv.org/abs/2607.15190)

**Authors**: Han Jiang, Sunbeom Kwon, Jinwen Luo, Ziang Xiao, Susu Zhang  
**Category**: cs.AI  
**Published**: 2026-07-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.15190v1  

#### Abstract
AI benchmarks increasingly leverage item-level statistical models, particularly item response theory (IRT), to estimate model capabilities, rank systems, select informative examples, and diagnose benchmark quality. However, AI benchmark data often departs from the data regime of human testing, for w...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Can We Trust Item Response Theory for AI Evaluation?**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决了什么问题  
本文系统性地探讨了 **Item Response Theory (IRT)** 在 **AI 模型评估** 中的可靠性问题。尽管 IRT 被广泛用于心理测量和人类测试，并被越来越多地应用于 AI 基准测试（如 LLM 排行榜），但其原始假设是为“大量受试者 + 少量题目”的人类测试场景设计的，而当前 AI 评估的数据模式恰恰相反：**少量模型（examinees） + 海量题目（items）**。

此外，AI 模型的能力分布常呈现 **非正态、偏斜、多峰** 特征，违背了 IRT 估计中常见的正态能力分布假设。因此，直接套用传统 IRT 工具可能导致参数估计偏差、排名失真等问题。

> 🔍 **核心问题**：我们能否信任在当前 AI 评估范式下应用的 IRT？它是否会产生误导性的结论？

---

### ✅ 提出了什么新方法或新思路  

本研究并未提出新的 IRT 模型，而是采用了一种 **基于真实数据校准的仿真研究框架（simulation study）** 来评估不同 IRT 估计器在典型 AI 评估条件下的表现。

#### 主要创新点包括：
- **首次系统性验证 IRT 在 AI 评估中的适用边界**，填补了该领域缺乏实证验证的空白。
- 构建了一个覆盖 **18,000 种仿真条件** 的大规模实验体系，涵盖六大数据集、三种主流 IRT 模型（1PL / 2PL / 3PL）、四种估计器（MML-EM / MCMC / VI / PSN）以及多个样本量水平。
- 使用从 OpenLLM Leaderboard 等真实基准提取的 **真实能力分布与题目参数作为“真值”**，使仿真更贴近实际 AI 评估场景。
- 明确识别出影响 IRT 可靠性的关键因素：**模型数量（N）、能力分布形态（尤其是偏度 skewness）、题目数量（J）**。

---

### ✅ 相比现有方法的优势  

| 方面 | 优势 |
|------|------|
| **方法论严谨性** | 遵循 ADEMP 框架，通过控制“已知真值”来评估估计质量，远优于仅在真实数据上观察相关性的经验性研究。 |
| **现实贴合度高** | 所有仿真参数均来源于真实 LLM 基准响应矩阵，避免理想化假设。 |
| **全面性与可复现性** | 公开实验设计细节（App. A）、评估指标定义、失败率统计等，便于后续研究复现与扩展。 |

---

## 2. **核心实验方法和设置**

### 📚 使用了哪些数据集  
实验基于 **OpenLLM Leaderboard v1** 中的六个代表性基准数据集构建仿真环境：

| 数据集 | 类型 | 题目数（J） | 模型数（N） |
|--------|------|-------------|------------|
| **ARC-CHALLENGE** | 常识推理 | ~644 | ~5,222 |
| **HellaSwag** | 情景补全 | ~5,711 | ~5,221 |
| **MMLU** | 多学科知识 | ~12,508 | ~5,219 |
| **TruthfulQA** | 真实性判断 | ~644 | ~5,222 |
| **WinoGrande** | 代词消解 | ~1,045 | ~6,550 |
| **GSM8K** | 数学推理 | ~1,306 | ~6,068 |

这些数据集涵盖了多种任务类型，且题目规模远超传统心理测验。

---

### ⚙️ 实验设置

#### （1）数据生成流程
1. 从上述数据集中预处理二元响应矩阵（0/1）；
2. 使用 **Variational Inference (VI)** 拟合 1PL/2PL/3PL 模型，获得“真实”的 item 参数（难度 $b$、区分度 $a$、猜测参数 $c$）和能力参数 $\theta$；
3. 在每种条件下（dataset × IRT model × sample size），随机抽取 N 个能力值（$\theta$），结合真实的 item 参数生成完整的响应矩阵；
4. 对每个配置进行 **50 次重复仿真**。

#### （2）评估的 IRT 估计器（共四种）
| 估计器 | 全称 | 特点 |
|-------|------|------|
| **MML-EM** | Marginal Maximum Likelihood via EM | 经典方法，假设能力服从正态分布；计算效率低 |
| **MCMC** | Markov Chain Monte Carlo | 贝叶斯采样，精度高但极慢 |
| **VI** | Variational Inference | 近似贝叶斯推断，速度快，广泛用于近期研究 |
| **PSN** | Pseudo-Siamese Network | 神经网络实现，不假设能力分布，适合大规模数据 |

#### （3）评估指标

| 用途 | 指标 |
|------|------|
| **计算可行性** | 成功率、运行时间、内存占用 |
| **能力估计可靠性** | Kendall’s τ（排名一致性）、Aggregate Score Error（平均得分误差） |
| **题目参数恢复** | Pearson 相关系数（difficulty/discrimination recovery）、Mean Item Error |
| **高效评估支持能力** | Fisher Information-based benchmark compression 效果（short-form ranking recovery） |

---

## 3. **主要实验结果和性能指标**

### 📊 关键性能数据汇总

| 结果维度 | 主要发现 |
|---------|----------|
| **计算可行性** | MML-EM 和 MCMC 在大题量（如 MMLU, J≈12k）下频繁因内存溢出或超时失败；VI 最快；PSN 可扩展性强但随 N/J 增长变慢。 |
| **模型排名恢复（Kendall’s τ）** | 当能力分布高度偏斜（skewness > 2.0）时，所有估计器的 τ 下降至 **< 0.6**；低偏度时可达 **> 0.85**。 |
| **题目参数恢复** | N=30 时几乎所有方法都严重失效（difficulty recovery < 0.5）；N≥100 后显著改善。 |
| **aggregate score error** | 多数情况下 < 0.025，但在小样本 + 3PL 下 VI 表现不稳定。 |
| **benchmark compression** | 所有估计器均优于随机选择，但提升幅度有限（margin modest），且不受 skewness 显著影响。 |

---

### 🔁 与基线方法的对比结果

| 估计器 | 排名恢复 | 题目参数恢复 | 计算效率 | 总体评价 |
|--------|----------|--------------|-----------|-----------|
| **MML-EM** | 中等 | 差（尤其小样本） | 极差（OOM） | ❌ 不推荐用于大型 AI benchmark |
| **MCMC** | 最佳 | 最佳（当可行时） | 极慢（>10k 秒） | ✅ 精度最高，但不可扩展 |
| **VI** | 较好 | 不稳定（部分条件下 difficulty recovery < 0.5） | ⚡ 最快 | ⚠️ 快速但需谨慎验证 item 参数 |
| **PSN** | 中等偏低 | 中等 | 良好（GPU 加速） | ✅ 适用于超大规模 benchmark，但无不确定性量化 |

> 💡 **重要发现**：**分布形状的影响远大于估计器选择** —— 即使是最好的估计器，在高度偏斜的能力分布下也无法恢复可靠排名。

---

### 🔍 消融实验结果（隐含分析）

虽然未明确标注“ablation”，但以下变量被系统操控并得出因果结论：

| 控制变量 | 发现 |
|--------|------|
| **样本量 N（30 vs 100 vs 1000）** | N=30 完全不足以恢复 item 参数；N≥100 是基本门槛；进一步增加收益递减。 |
| **能力分布偏度（skewness）** | 是决定 ranking recovery 的最关键因素，高于估计器选择和样本量。 |
| **题目数量 J** | 导致 MML/MCMC 内存崩溃，对 VI/PSN 影响较小。 |
| **IRT 模型复杂度（1PL→3PL）** | 3PL 极难收敛，尤其在小样本下几乎不可行。 |

---

## 4. **关键结论和发现**

### ✅ **主要发现**

1. **当前 AI 评估范式严重挑战 IRT 的有效性前提**：
   - 小模型池（N < 100）、巨量题目（J > 10k）、非正态能力分布共同导致传统 IRT 工具可能产生 **误导性结论**。

2. **样本量至关重要**：
   - **N=30 是不可接受的小样本**，无法支持可靠的 item-level 分析；
   - **N≥100 是进行 item parameter recovery 的最低建议门槛**。

3. **能力分布形态主导排名可靠性**：
   - 高度偏斜或多峰的能力分布会导致 ceiling/floor effect，使得大多数题目失去区分力，从而破坏模型排名。

4. **估计器各有优劣，需按场景选择**：
   - 若追求精度且资源充足 → **MCMC**
   - 若追求速度且样本适中 → **VI**（但需警惕 item 参数偏差）
   - 若处理超大规模 benchmark → **PSN**
   - **MML-EM 应避免用于现代 AI benchmark**

5. **IRT 支持 benchmark compression 是稳健的**：
   - 即使参数估计存在偏差，基于 Fisher Information 的 item selection 仍能优于随机抽样，说明粗粒度的信息可用于压缩。

---

### ⚠️ **方法的局限性**

| 局限 | 说明 |
|------|------|
| **硬件异构性** | MML/MCMC/Vi 在 CPU 上运行，PSN 在 GPU 上，运行时间不具备完全可比性。 |
| **未测试中间样本量** | 缺少 N=50~90 的结果，难以精确界定“足够样本量”。 |
| **仅考虑单维 IRT** | 忽略了多维能力结构（如不同任务维度）的影响。 |
| **未整合语义信息** | 仅依赖响应矩阵，忽略了 item 内容本身的质量信号。 |
| **收敛监控不足** | 仅对 MML-EM 监控收敛，其他方法默认成功。 |

---

### 🔮 **未来工作方向**

1. 开发更适合 AI 评估范式的新型 IRT 框架，例如：
   - 自适应先验（adaptive priors）以应对非正态能力分布；
   - 正则化策略缓解小样本过拟合；
   - 图神经网络融合 item 文本特征。

2. 探索更高效的 scalable estimators，结合神经估计与统计保证。

3. 研究如何利用 IRT 指导动态 benchmark 构建与演化测试（evolving testing）。

4. 建立 **IRT 应用指南工具包**，帮助研究者根据自身数据特征判断是否适用 IRT 及应选用何种估计器。

---

## ✅ 总结一句话

> **在当前 AI 评估的小样本、大题量、非正态能力分布背景下，盲目使用传统 IRT 工具存在重大风险；必须结合仿真验证、提高样本量（N≥100）、谨慎选择估计器，并优先关注能力分布形态对结果可信度的根本影响。**

</details>

---

### 16. [SEED: Self-Evolving On-Policy Distillation for Agentic Reinforcement Learning](https://arxiv.org/abs/2607.14777)

**Authors**: Jinyang Wu, Shuo Yang, Zhengxi Lu, Fan Zhang, Yuhao Shen, Lang Feng, Haoran Luo, Zheng Lian, Shuai Zhang, Zhengqi Wen, Jianhua Tao  
**Category**: cs.CL  
**Published**: 2026-07-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.14777v1  

#### Abstract
Large language models are increasingly trained as interactive agents for long-horizon tasks involving multi-turn interaction, tool use, and environment feedback. Outcome-based reinforcement learning (RL) provides a practical optimization paradigm, but its sparse trajectory-level rewards offer limite...

---

### 17. [Latent Communication Between Language Model Agents: Channels, Alignment, and the Limits of Text](https://arxiv.org/abs/2607.14103)

**Authors**: Markus Wenzel  
**Category**: cs.CL  
**Published**: 2026-07-17  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.14103v1  

#### Abstract
Multi-agent systems (MAS) are utilized in many contexts and many professions. Those MAS rely on inter-agent communication, usually implemented by clear-text message passing. We hypothesize that Large Language Models may have a world model at their disposal that exceeds expressibility in text when co...

---

### 18. [Enhancing Small Language Models Reasoning through Knowledge Graph Grounding](https://arxiv.org/abs/2607.14149)

**Authors**: Dimitrios Kelesis, Konstantinos Bougiatiotis, Georgios Paliouras  
**Category**: cs.AI  
**Published**: 2026-07-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.14149v1  

#### Abstract
Although large language models (LLMs) have set benchmarks for zero-shot reasoning, their deployment remains cost-prohibitive and environmentally taxing. Small Language Models (SLMs) offer a sustainable alternative, but prone to errors, on tasks requiring complex, multi-hop logical grounding. We inve...

---

### 19. [MemoHarness: Agent Harnesses That Learn from Experience](https://arxiv.org/abs/2607.14159)

**Authors**: Yue Huang, Wenjie Wang, Han Bao, Yuchen Ma, Xiaonan Luo, Yi Nian, Haomin Zhuang, Zheyuan Liu, Yue Zhao, Xiangliang Zhang  
**Category**: cs.AI  
**Published**: 2026-07-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.14159v1  

#### Abstract
An agent harness is the external control layer that turns a base LLM into an executable agent by managing context, tools, orchestration, memory, decoding, and output handling. While harness design strongly affects agent behavior, most automatic improvement methods optimize narrower artifacts such as...

---

### 20. [CFM-Bench: A Unified Multi-Domain, Multi-Task Benchmark for Channel Foundation Models](https://arxiv.org/abs/2607.14975)

**Authors**: Yuan Gao, Wenjun Yu, Jun Jiang, Yunfan Li, Xinyu Guo, Shugong Xu  
**Category**: cs.AI  
**Published**: 2026-07-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.14975v1  

#### Abstract
Channel foundation models (CFMs) are developing rapidly, with recent studies reporting benefits from pretraining across downstream wireless tasks. Yet CFMs are commonly evaluated in model-specific pipelines with different data, radio configurations, partitions, adaptation procedures, task definition...

---

### 21. [SearchOS-V1: Towards Robust Open-Domain Information-Seeking Agent Collaboration](https://arxiv.org/abs/2607.15257)

**Authors**: Yuyao Zhang, Junjie Gao, Zhengxian Wu, Jiaming Fan, Jin Zhang, Shihan Ma, Yao Yao, Weiran Qi, Chuyan Jin, Guiyu Ma, Xingzhong Xu, Kai Yang, Ji-Rong Wen, Zhicheng Dou  
**Category**: cs.AI  
**Published**: 2026-07-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.15257v1  

#### Abstract
Recent advances in Tool-Integrated Large Language Models have made web search a core capability of information-seeking agents. However, as interaction histories grow, agents increasingly struggle to track task progress. When search attempts fail to yield useful evidence, current single- and multi-ag...

---

### 22. [Long-term User Engagement Optimization through Model-agnostic Downstream Rewards Learning](https://arxiv.org/abs/2607.14192)

**Authors**: Dingsu Wang, Filip Ryzner, Kelly He, Armando Ordorica, David Woo, Aditya Mantha, Liyao Lu, Usha Amrutha Nookala, Haoran Guo, Jiacong He, Olafur Gudmundsson, Matt Chun, Krystal Benitez, Dhruvil Deven Badani, Yijie Dylan Wang  
**Category**: cs.LG  
**Published**: 2026-07-17  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.14192v1  

#### Abstract
As recommender systems mature in the past few years, their optimization objectives have evolved from a primary focusing on short-term behavioral signals to a broader emphasis on long-term user engagement and retention. However, directly optimizing retention is difficult because return signals are sp...

---

### 23. [Augmentations for Robust and Efficient Imitation Learning in Streamed Video Games](https://arxiv.org/abs/2607.14200)

**Authors**: Somjit Nath, Abdelhak Lemkhenter, Pallavi Choudhury, Chris Lovett, Katja Hofmann, Sergio Valcarcel Macua, Lukas Sch\"afer  
**Category**: cs.LG  
**Published**: 2026-07-17  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.14200v1  

#### Abstract
Imitation learning is an appealing way to scale game-playing agents to complex 3D environments by training policies to map visual observations to actions from human demonstrations. However, these demonstrations are expensive to collect and modern game-playing is often done through streaming in which...

---

### 24. [Lyapunov Guidance: A Unified Framework for Stabilizing Generative Flows](https://arxiv.org/abs/2607.14272)

**Authors**: Jingdong Zhang, Xinze Li, Yize Jiang, Luan Yang, Minkai Xu, Junhong Liu  
**Category**: cs.LG  
**Published**: 2026-07-17  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.14272v1  

#### Abstract
Flow matching has emerged as an effective framework for learning complex data distributions, but adapting pretrained flow models to new tasks often requires computationally expensive retraining. Post-training guidance provides a more efficient alternative, but existing methods are largely heuristic ...

---

### 25. [A Noise-Robust Elicit-to-Optimize Framework for Distortion Riskmetrics via Inverse Reinforcement Learning](https://arxiv.org/abs/2607.14373)

**Authors**: Yang Liu, Yuhao Liu, Yunran Wei  
**Category**: cs.LG  
**Published**: 2026-07-17  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.14373v1  

#### Abstract
We propose a noise-robust elicit-to-optimize framework that integrates inverse reinforcement learning (IRL) and reinforcement learning (RL) for eliciting agents' risk preferences and optimizing policies under a broad class of risk objectives characterized by distortion riskmetrics. On the elicitatio...

---

### 26. [Active Real-World Factor-Based Evaluation for Generalist Robot Policies](https://arxiv.org/abs/2607.14439)

**Authors**: Andrew Liao, Hanchen Cui, Karthik Desingh, Aryan Deshwal  
**Category**: cs.LG  
**Published**: 2026-07-17  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.14439v1  

#### Abstract
Generalist robot manipulation policies trained on large, diverse datasets have shown remarkable promise across a wide range of tasks. However, rigorously evaluating these policies remains a fundamental challenge. Real-world performance depends on a large combinatorial space of task factors including...

---

### 27. [Interleaved Noise Injection Improves Clean, Corrupted, and OOD Performance](https://arxiv.org/abs/2607.14466)

**Authors**: Matt L. Wiemann, Peter Melchior, Andrew K. Saydjari  
**Category**: cs.LG  
**Published**: 2026-07-17  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.14466v1  

#### Abstract
Noise injection is a well-known technique in stochastic optimization. We report its surprising effectiveness with an interleaved (on-off-on-off...) rather than the usual monotonic decay schedule. We present a theoretical analysis of noise injection, which confirms that corruption by impulse noise ap...

---

### 28. [HG-RAG: Hierarchy-Guided Retrieval-Augmented Generation for Structured Knowledge Graphs](https://arxiv.org/abs/2607.14095)

**Authors**: Pranav Yadav  
**Category**: cs.AI  
**Published**: 2026-07-17  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.14095v1  

#### Abstract
Retrieval Augmented Generation (RAG) has proven to be a widely successful process at improving the quality of outputs from a Large Language Model (LLM) for wider context. However, RAG systems typically retrieve context from flat document stores, which struggles when queries require hierarchical or r...

---

### 29. [RegNetAgents: A Multi-Agent Framework for Cross-Network Regulatory Driver Identification in Cancer Genomics](https://arxiv.org/abs/2607.14097)

**Authors**: Jose A. Bird  
**Category**: cs.AI  
**Published**: 2026-07-17  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.14097v1  

#### Abstract
We introduce RegNetAgents, an AI-oriented multi-agent framework for structured, query-driven regulatory candidate identification across heterogeneous gene regulatory networks. The system enables unified analysis of bulk tumor and single-cell-derived ARACNe networks by integrating TCGA-derived cancer...

---

### 30. [Capability from Access Structure, Not Scale: Lower Bounds and Pre-Registered Tests for Hybrid Sequence Models](https://arxiv.org/abs/2607.14144)

**Authors**: Wenhui Chen, Jianlin Chen, Ziyao Lin, Chi Man Vong  
**Category**: cs.AI  
**Published**: 2026-07-17  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.14144v1  

#### Abstract
The Platonic Representation Hypothesis (PRH) holds that as models scale, representations of heterogeneous networks converge toward a shared model of reality. We propose its sequel and boundary, the Capability Convergence Hypothesis (CCH): under a fixed per-token inference budget, representational co...

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

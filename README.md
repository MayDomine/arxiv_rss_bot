# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-09-22 10:21:51 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Adapting Tree-Structured Speculative Decoding to DeepSeek-V4 for Efficient Inference](https://arxiv.org/abs/2609.24698)

**Authors**: Changxu Liu, Zhaogeng Li  
**Category**: cs.CL  
**Published**: 2026-09-22  
**Score**: 13.0  
**Type**: new  
**ArXiv ID**: 2609.24698v1  

#### Abstract
Repeated execution of the target model during autoregressive decoding is a major source of LLM inference latency. Unlike linear speculation, which follows a single candidate chain, tree-structured speculation retains multiple branches from shared prefixes; under the same budget, this broader coverag...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Adapting Tree-Structured Speculative Decoding to DeepSeek-V4 for Efficient Inference*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **DeepSeek-V4** 这类采用 **CSA/HCA online compressed attention** 架构的模型中，传统的 **tree-structured speculative decoding** 面临严重挑战。由于其注意力机制在线压缩上下文历史（context history），不同候选分支在分叉后会生成不同的压缩状态，导致跨分支的 **state inconsistency**，破坏验证过程的正确性。

因此，直接将 tree speculation 应用于 DeepSeek-V4 会导致：
- 分支间状态污染（rejected branch 的压缩状态影响后续解码）
- 验证逻辑不一致（causal visibility 错误）
- 接受路径无法安全刷新

该问题在传统 dense attention 模型中并不存在，但在 **compressed/sparse/structured context 表示** 日益普及的背景下变得关键。

---

### 🚀 提出的新方法
作者提出了一套完整的适配方案，使 **tree-structured speculative decoding** 能够在 DeepSeek-V4 上高效、正确地运行，核心创新包括：

#### （1）**Branch-aware causal verification**
- 在 target verify 阶段引入树感知的 attention mask，确保每个候选 token 只能访问其祖先路径上的 token，而非其他分支的内容。
- 维护正确的 **causal dependency topology**，避免信息泄露。

#### （2）**Temporary state isolation via scratch pad**
- 所有 speculative branch 的 KV states 和 CSA/HCA 压缩中间状态均暂存于临时 **scratch pad** 中，不写入持久化 cache。
- 防止 speculative 更新提前污染主 context。

#### （3）**Accepted-path state refresh**
- 验证完成后，仅将被接受路径的状态“刷新”回主 cache。
- 包括 token history、KV cache、CSA/C4、HCA/C128 的压缩状态及中间 buffer。
- 实现端到端的 **state consistency**。

#### （4）系统级优化
- **Scratch-pad execution** 减少持久 cache 写入开销
- **Device-side metadata processing** 提升执行重叠
- **CUDA Graph 覆盖控制** 适应动态树结构
- **C4/C128 差异化 refresh schedule**：高频压缩路径及时更新，低频路径延迟合并写入

---

### 🔍 相比现有方法的优势

| 对比维度 | Linear Speculation | Tree Speculation (本文) |
|--------|------------------|----------------------|
| 候选组织 | 单链结构，错误传播严重 | 多分支共享前缀，容错性强 |
| 接受长度 | 易因早期错误截断 | 更大概率找到长接受路径 |
| 适配难度 | 无需处理分支隔离 | 需解决压缩状态一致性 |
| 吞吐提升潜力 | 有限 | 在合适配置下可达 +18.5% |

> ⚠️ 特别指出：**tree speculation 的优势集中在 verify side**，与 DSpark 等 draft-side 方法正交且可组合。

---

## 2. 核心实验方法和设置

### 📚 数据集
使用三个具有不同生成特性的任务进行评估：
- **GSM8K**：数学推理题，逻辑较规则、可预测性高
- **MBPP**：Python 编程任务，中等不确定性
- **ShareGPT**：开放域多轮对话，内容发散、不可预测性强

> 目的是观察 tree speculation 在从“高度可预测”到“高度发散”负载下的表现差异。

---

### ⚙️ 实验设置

| 参数 | 设置 |
|-----|------|
| 模型 | DeepSeek-V4-Flash |
| 硬件 | 8-GPU NVIDIA 机器 |
| 验证预算 D | 5, 6, 7, 8（即每轮验证最多 D 个候选 token） |
| Batch Size | 1, 2, 4, 8, 16, 32, 64 |
| Draft Steps / top-k | 控制变量：<br>- Linear: `sN_k1_dD`（如 `s4_k1_d5`）<br>- Tree: `sN_k2_dD`（如 `s4_k2_d5`） |
| 控制变量原则 | 同一 D 下，linear 与 tree 验证相同数量的 token，形成公平对比 |

---

### 📊 评估指标

| 指标 | 定义 | 用途 |
|------|------|------|
| **Accepted Length** | 每轮 speculative decoding 平均接受的 token 数量 | 衡量候选质量与接受效率 |
| **Decode Throughput** | 单位时间内完成的有效解码量（相对提升） | 衡量端到端性能增益 |

> 注：所有结果以 **matched linear configuration** 为基准（0%线），向上表示优于基线。

---

### 🆚 基线方法对比
- **Baseline**: Linear speculative decoding（top-k=1）
- **Proposed**: Tree-structured speculation（top-k=2），相同验证预算 D
- 所有比较均为 **controlled experiment**，唯一变量是候选组织方式（chain vs. tree）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

| 配置 | Avg Accepted Length (D=8) | Throughput Gain |
|------|----------------------------|-----------------|
| Linear (k=1) | ~2.84 (GSM8K), ~2.68 (MBPP), ~2.39 (ShareGPT) | 0%（基准） |
| Tree (k=2)   | ~3.41 (GSM8K), ~3.16 (MBPP), ~2.84 (ShareGPT) | **最高 +18.5%** |

> 示例：在 `D=8`, `bs=4`, `ShareGPT` 上达到峰值增益 **+18.5%**

---

### 🔁 与基线方法的对比结果

#### （1）Accepted Length 全面领先
- 在所有 D ∈ {5,6,7,8}、所有 batch size、所有数据集上，**tree 始终优于 linear**
- 提升幅度随 D 增加而单调上升：
  - D=5: +11.0%
  - D=6: +14.4%
  - D=7: +17.1%
  - D=8: **+18.6%**

> 表明更大的预算允许更宽的树结构发挥优势。

#### （2）Throughput 改进显著（除极小预算外）
- D=5 时增益微弱（约 +3–5%），部分配置接近 break-even
- D≥6 时增益稳定在 **+8–10%**
- 最高吞吐提升达 **+18.5%**（s3_k2_d6, bs=4, ShareGPT）

> 增益 = 接受长度提升 − tree 引入的额外开销（metadata、state isolation、refresh）

#### （3）Accepted Length 几乎与 batch size 无关
- 在 bs=1 到 bs=64 范围内波动 < 0.04
- 说明接受长度由 draft quality 和 candidate structure 决定，非执行层因素

#### （4）Throughput 呈现 “倒U型” 与 batch size 关系
- 增益在 **bs=4 左右达到峰值**
- 随着 bs 增大至 64，增益下降（如 D=6 时从 +11% → +7%）
- 原因：
  - 小 bs：memory-bound，减少 target forward pass 效果明显
  - 大 bs：compute-bound，额外 candidate 争抢计算资源

#### （5）越难预测的任务，收益越大
- ShareGPT（最难预测）> GSM8K ≈ MBPP
- 因为单链更容易早期失败，tree 的宽度能有效恢复浪费的 speculative effort

---

### 🔍 消融分析（隐含在实验趋势中）

虽然未设显式消融实验，但从以下现象可反推组件重要性：

| 观察 | 推论 |
|------|------|
| 小预算（D=5）增益微弱 | 表明 overhead 显著，**temporary state management 和 metadata processing 成本不可忽略** |
| 大预算下 accepted length 持续增长但 throughput plateau | 表明 **overhead 随树复杂度增长**，存在收益饱和点 |
| shallow-and-wide 树优于 deep-and-narrow | 表明当前 draft model 不足以支撑长距离可靠预测，**width 更有价值** |

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Tree-structured speculation 可成功适配 DeepSeek-V4**
   - 通过 **branch-aware verification + temporary state isolation + accepted-path refresh** 三阶段机制，解决了 CSA/HCA 压缩注意力带来的 state consistency 挑战。

2. **在相同验证预算下，tree 始终获得更高的 accepted length**
   - 平均比 linear 多接受 **~0.5–0.6 个 token/轮次**（D=8 时）
   - 提升随预算增大而增加，最大达 **+18.6%**

3. **Throughput 提升可达 +18.5%，尤其适合特定场景**
   - 最佳条件：**D ≥ 6**, **small-to-medium batch size (e.g., bs=4)**, **low-predictability workloads (e.g., ShareGPT)**

4. **Accepted length 与 throughput 解耦**
   - 当 D > 6–7 时，accepted length 继续上升，但 throughput 增益趋于 plateau
   - 原因：tree 的 overhead（state isolation、refresh）增长抵消了更多接受 token 的好处

5. **Tree speculation 与 draft-side 方法（如 DSpark）正交**
   - 本文聚焦 verify side，DSpark 聚焦 draft side
   - 二者可组合，未来有望叠加增益

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| Overhead 显著 | 在小预算（D=5）或大 batch size 下难以体现优势 |
| 依赖 tree shaping 策略 | 当前使用固定 top-k=2，未动态调整深度/宽度 |
| 未探索硬件感知 budget tuning | CUDA occupancy、memory bandwidth 等未精细优化 |
| 仅验证 FlashMLA 路径 | 更复杂的 draft policy（如 EAGLE-2 动态树）尚未集成 |

---

### 🔮 未来工作方向

1. **Adaptive, workload-aware tree shaping**
   - 根据 runtime workload predictability 和 batch size 动态决定是否启用 tree 及其宽度
   - 结合 draft confidence、path structure、inter-candidate dependency 进行智能剪枝

2. **Joint depth-width optimization**
   - 联合优化 draft-side depth 与 verify-side width，实现 total cost 最小化

3. **Extend to other compressed attention paradigms**
   - 如 DeepSeek-V4.1-Flash（CSA2）、其他 hierarchical/sparse context models

4. **Co-design with draft models**
   - 训练能输出 tree structure 的 draft model，进一步提升 accepted length

5. **Hardware-aware budget scheduling**
   - 基于 GPU occupancy 自动选择最优 D 和 tree shape

---

## 总结

> **Tree-structured speculative decoding 在 DeepSeek-V4 上是可行且高效的，但其价值取决于 verify-side 的适配能力。**

本文首次系统解决了 **compressed attention 模型中的 tree speculation state consistency 问题**，并通过工程优化实现了高达 **+18.5% 的 decode throughput 提升**。更重要的是，它揭示了一个趋势：

> 随着 draft models 的进步，**瓶颈正在从 draft capability 转向 verify-side efficiency**；  
> 而随着越来越多模型采用 **structured context representations（如 CSA/HCA）**，  
> **verify-side 的适配复杂度将成为 speculative decoding 是否能落地的关键。**

因此，**target-verify adaptation 是未来 LLM inference acceleration 的核心战场之一**。

</details>

---

### 2. [NAVIR: Neuromorphic Audio-Visual Speech Recognition for Robust Human-Robot Interaction on Edge Hardware](https://arxiv.org/abs/2609.24391)

**Authors**: Leonidas Delimpasis, Panagiota Moraiti, Antonis Porichis, Panos Chatzakos, Michail Karamousadakis  
**Category**: cs.LG  
**Published**: 2026-09-22  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2609.24391v1  

#### Abstract
Voice-controlled interaction in industrial settings is hampered by acoustic noise, which severely degrades audio-only speech recognition. Audio-visual speech recognition (AVSR) addresses this by fusing lip-motion cues with the audio stream, but state-of-the-art pipelines rely on three-dimensional co...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《NAVIR: Neuromorphic Audio-Visual Speech Recognition for Robust Human-Robot Interaction on Edge Hardware》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在工业环境中，**高噪声环境**严重削弱了传统基于音频的自动语音识别（ASR）系统的性能，导致人机交互不可靠。尽管**Audio-Visual Speech Recognition (AVSR)** 能通过融合唇部运动信息提升鲁棒性，但主流高性能 AVSR 模型依赖于 **3D 卷积、循环结构（如GRU）、注意力机制**等计算密集型组件，难以部署在资源受限的边缘设备上。

此外，现有研究尚未在**类脑神经形态硬件**（neuromorphic hardware）上实现完整的多模态 AVSR 系统。

### 提出的新方法与创新点
本文提出了 **NAVIR** —— 一个端到端运行在 **BrainChip Akida neuromorphic 处理器** 上的 AVSR 系统，其核心创新如下：

- ✅ **硬件兼容架构设计**：  
  针对 Akida 芯片仅支持二维卷积、不支持 3D 卷积、循环层或注意力机制的限制，提出将空间编码与时间编码解耦为独立模块：
  - **Per-frame 视觉编码器**（AkidaNet）
  - **Temporal 视频编码器**（跨帧时序建模）
  - **Spectrogram 音频编码器**
  - 所有模块均满足 Akida 的硬件约束。

- ✅ **轻量级预测头与语法约束解码器**：  
  使用 **MLP Predictor Head** 进行模态融合，并引入 **Constrained Beam Search Decoder**，利用任务固定的语法规则限制搜索空间，保证输出语法合法且显著降低推理开销。

- ✅ **量化感知训练（Quantization-Aware Training, QAT）策略**：  
  采用混合精度量化方案（8/4/4 用于输入编码器，4/4/4 用于中间模块），并在训练后期进行 QAT，以恢复量化带来的精度损失。

- ✅ **首次完整部署于神经形态硬件的多模态 AVSR 系统**：  
  据作者所知，这是第一个在 Akida 类神经形态芯片上实现从原始音视频输入到文本输出的完整 AVSR 流水线。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **能效** | 在 spiking formulation 下理论能耗比传统 ANN 低 **13.17×**；实测比 Raspberry Pi CPU 低约 **5×**，比笔记本 GPU 低 **超100×** |
| **实时性** | 在 Akida 上实现 **14.5 inferences/sec**，满足命令级实时响应需求 |
| **鲁棒性** | 在噪声环境下，AV 融合模型显著优于纯音频模型 |
| **可部署性** | 完全适配边缘设备，无需云端依赖 |

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 描述 |
|-------|------|
| **GRID Corpus** | 公共基准数据集，包含 34 名说话者，每人 1,000 条句子，遵循固定六词语法（command-color-preposition-letter-digit-adverb）。用于标准评估。 |
| **NAVIR Industrial-Command Corpus (内部)** | 自建工业机器人控制指令数据集，含 183 条命令，涵盖“移动”、“拾取”、“放置”、“前往”、“旋转”五类动作，由两名说话者录制，共 366 条样本。更贴近实际应用场景。 |

### 实验设置
- **预处理**：
  - 视频：使用 MediaPipe Face Mesh 提取唇部区域（32×64 或 88×176）
  - 音频：转换为 MFCC 特征（Mel bands=112）
  - 对齐窗口：视频与音频滑动窗中心对齐，确保时间同步
- **训练策略**：
  - 使用 **Connectionist Temporal Classification (CTC)** 损失函数
  - 引入 **UrbanSound8K** 中的机械噪声（空调、钻孔、怠速引擎、电锤）进行数据增强，SNR 设置为 {-15, -10, -5, 0} dB
  - 先浮点训练，后进行 **QAT 微调**
- **量化配置**：
  - 图像/音频编码器：8/4/4（weight_in/weight/activation）
  - 视频编码器与预测头：4/4/4

### 评估指标
- **Word Error Rate (WER)**：主要评价指标
- **Sentence-Level Command Accuracy**：针对 NAVIR 数据集的任务级准确率
- **Energy Consumption per Inference**：实测功耗（mWh/inference）
- **Throughput (inferences/sec)**

### 基线方法对比
- **Audio-only models**（clean/noisy training）
- **Video-only models**
- **Fused AV models**
- 与文献中 SOTA lip-reading 模型（如 LipNet、Wu et al. 2024）在 WER 和 FLOPs 上进行 Pareto 分析

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Quantized 模型）

#### 在 **GRID Corpus** 上的结果：
| 模型 | 测试条件 | Unseen-Speaker WER | Overlapped-Speaker WER |
|------|----------|---------------------|-------------------------|
| Audio-only (noisy) | Noisy Audio | 22.5% | 11.8% |
| Video-only | — | 35.3% | 6.7% |
| **AV Fusion (proposed)** | **Noisy Audio** | **14.0%** | **3.3%** |
| AV Fusion (clean) | Clean Audio | 5.3% | 0.8% |

> 🔍 **分析**：在噪声条件下，AV 融合相比纯音频模型分别降低 **8.5%** 和 **8.5%** 的 WER，体现视觉模态的有效锚定作用。

#### 在 **NAVIR Corpus** 上的结果：
| 模型 | Clean Audio WER / Acc | Noisy Audio WER / Acc |
|------|------------------------|------------------------|
| Audio-only | 6.6% / 91.5% | 98.7% / 0.0% |
| Video-only | 0.7% / 100.0% | 0.7% / 100.0% |
| **AV Fusion (proposed)** | **0.6% / 98.6%** | **1.5% / 98.6%** |

> ✅ 在真实工业命令场景下，AV 模型达到 **98.6% 命令准确率** 和仅 **1.5% WER**，即使在强噪声下仍保持高可靠性。

### 与基线方法对比
- 在 **GRID** 上，虽然 video-only 模型 WER（35.3%）高于 SOTA（~10%），但其 **FLOPs 仅为 LipNet 的 1/3.1，参数量少 3×以上**，体现了在极低算力下的合理折衷。
- 在 **Pareto Frontier 分析**（图2）中，NAVIR 是唯一处于 **<3 GFLOPs 区域** 的模型，适合边缘部署。

### 消融实验结果（隐含分析）
- **QAT 效果**：部分配置下量化后性能反而略有提升（如 overlapped 视频模型从 9.1% → 6.7%），表明 QAT 有助于适应硬件特性。
- **噪声增强必要性**：未在噪声数据上训练的模型在噪声测试中 WER 飙升至 ~77–80%，验证了数据增强的关键作用。
- **模态互补性**：单独任一模态失败时（如音频崩溃），另一模态仍可维持基本功能。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **多模态融合显著提升抗噪能力**：在工业噪声下，视觉信息有效补偿音频退化，使 WER 显著下降。
2. ✅ **神经形态硬件可用于复杂多模态任务**：首次证明 SNN 可在 Akida 上高效执行 AVSR，打破“仅适用于简单分类”的刻板印象。
3. ✅ **解耦式时空编码是可行路径**：即便缺乏 3D 卷积和注意力机制，通过分阶段建模仍可获得可用性能。
4. ✅ **极高能效比**：
   - 理论分析显示 SNN 相比 ANN 有 **13.17× 能效增益**
   - 实测显示比 CPU 低 **5×**，比 GPU 低 **超100×** 能耗
5. ✅ **端到端闭环验证成功**：系统已集成至 uFactory xArm 6 机械臂，实现“语音 → 动作”闭环控制。

### 方法的局限性
1. ❌ **受限于 Akida 架构**：无法使用 3D Conv、RNN、Attention，导致在复杂 lip-reading 任务上性能不及 SOTA。
2. ❌ **音频-视频模型吞吐下降**：由于工具链映射问题（见 Appendix A），audio-video 模型需更多硬件上下文切换，导致推理速度低于 video-only 模型。
3. ❌ **数据集规模有限**：NAVIR 仅有两个说话者，缺乏跨说话人泛化能力验证。
4. ❌ **前端仍在 CPU 运行**：MediaPipe Face Mesh 未部署在 Akida 上，仍是非神经形态组件。

### 未来工作方向
1. ✅ 探索 **sparsity-aware fine-tuning** 或 **magnitude pruning** 优化权重稀疏性，改善硬件映射效率。
2. ✅ 将 **face landmark detector 蒸馏为 Akida-compatible CNN**，实现全流程神经形态加速。
3. ✅ 扩展 **NAVIR Corpus** 至更多说话人、更大词汇量和多样化噪声场景。
4. ✅ 利用 Akida 的 **on-chip edge learning** 能力实现用户个性化自适应。
5. ✅ 迁移至下一代芯片 **AKD1500**，有望进一步降低静态功耗并支持更复杂拓扑。

---

> 📌 **总体结论**：  
> NAVIR 成功展示了在严格硬件约束下构建鲁棒、高效、可部署的多模态语音识别系统的可行性，为工业级人机交互提供了一条面向边缘计算与神经形态硬件的新范式。尽管在绝对精度上尚未超越 SOTA，但在 **能效、实时性和实用性** 方面取得了突破性进展。

</details>

---

### 3. [ARM: Attention with Routed-Memory for Learnable Sparse Control](https://arxiv.org/abs/2609.24417)

**Authors**: Qiuhao Zeng, Jerry Huang, Peng Lu, Ruiyi Fang, Gezheng Xu, Zihao Jing, Yufei Cui, Charles Ling, Gang Niu, Boyu Wang  
**Category**: cs.LG  
**Published**: 2026-09-22  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2609.24417v1  

#### Abstract
Despite advances in long-context inference, large language models (LLMs) remain fundamentally limited by the key-value (KV) caching mechanisms that are necessary for stable computation. Techniques such as selective token eviction and pruning have vastly mitigated these issues, but often discard core...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：ARM: Attention with Routed-Memory for Learnable Sparse Control**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
大型语言模型（LLMs）在长上下文推理中面临 **KV-Cache**（Key-Value Cache）机制的根本限制：
- 随着上下文长度增长，KV-Cache 的大小线性膨胀，导致：
  - **内存占用过高**（超出GPU容量）
  - **解码延迟增加**（注意力计算复杂度为 $O(T^2)$）
  - **信息丢失风险**：传统剪枝（pruning）、滑动窗口等策略会硬性丢弃历史token，可能移除关键信息。

现有方法如量化（quantization）、缓存卸载（offloading）、选择性保留（selective retention）虽缓解问题，但仍存在以下缺陷：
- 无法动态适应不同任务的信息需求；
- 固定规则导致次优决策；
- 卸载带来CPU-GPU传输瓶颈。

---

### **提出了什么新方法或新思路**
本文提出 **Attention with Routed-Memory (ARM)**，一种全新的可学习稀疏控制KV缓存架构，核心思想是将KV-Cache建模为一个**固定大小、层次化路由的可微分记忆系统**。

#### **三大创新组件**：

1. **Learnable Soft Eviction（可学习软淘汰）**
   - 使用 **Gumbel-Softmax** 实现端到端可微的slot选择。
   - 引入 **Sigmoid-Gated Update** 机制，在选定slot中“软融合”新旧KV对，避免硬覆盖。
   - 公式示例：
     $$
     \mathbf{A}_{t,y_t} = (1-\gamma_t)\cdot\mathbf{A}_{t-1,y_t} + \gamma_t\cdot(\text{new value})
     $$
     其中 $\gamma_t$ 是由当前token决定的更新门控。

2. **Adaptive Top-M Retrieval（自适应Top-M检索）**
   - 将检索预算 $M$ 的选择建模为一个 **Markov Decision Process (MDP)**。
   - 策略网络基于输入上下文动态决定应访问多少个memory bucket。
   - 支持简单查询用少量memory，复杂推理调用更多资源，实现**输入感知的稀疏性**。

3. **Hierarchical Router Structure（层次化路由器结构）**
   - KV-Cache组织成一棵多层树，叶子节点为memory bucket。
   - 路由器函数引导每个token进入语义相似的bucket，提升局部性和聚类效率。
   - 总体保持**恒定内存足迹**，不随序列增长而扩展。

---

### **相比现有方法的优势**
| 维度 | 传统方法 | ARM |
|------|--------|-----|
| 内存增长 | 线性增长或需外部存储 | 固定大小，完全驻留GPU |
| 信息保留 | 易因硬淘汰丢失关键信息 | 软融合保留历史信号 |
| 检索模式 | 固定Top-k或全量 | 动态调整，按需访问 |
| 可训练性 | 多为启发式规则 | 完全端到端可微，联合优化 |
| 推理效率 | 卸载高延迟，滑窗受限 | 低延迟，支持超长上下文 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **通用常识与语言建模基准**：
  - `WikiText`, `LAMBADA`（语言建模）
  - `PIQA`, `HellaSwag`, `Winogrande`, `SIQA`, `BoolQ`, `ARC`（常识推理）

- **短上下文回忆密集型任务**（recall-intensive）：
  - `FDA`, `SWDE`, `SQuAD`, `TriviaQA`, `Natural Questions`, `DROP`

- **长上下文理解基准**：
  - **LONGBENCH**（Bai et al., 2024）：涵盖单/多文档问答、摘要、代码补全等，最长达 **128K tokens**。
  - **RULER**（Hsieh et al., 2024）：系统测试模型在不同context length（4K–128K）下的真实能力。

---

### **实验设置和评估指标**
- **模型基础**：Llama3-8B
- **预训练数据**：FineWeb-Edu 子集（10B tokens）
- **KV槽位配置**：256个slots（与Sliding Window等基线一致），构建为4层树结构（每节点4子节点 → $4^4=256$）
- **评估指标**：
  - 常识/问答任务：Accuracy / F1
  - 语言建模：Perplexity (PPL)
  - 长文本任务：各子任务得分及平均分
  - 效率指标：**KV-Cache内存占用**、**生成延迟（latency）**

---

### **基线方法对比**
| 基线 | 类型 | 描述 |
|------|------|------|
| Full Attention | 上限参考 | 使用FlashAttention维护完整KV-Cache |
| SWA(256) | 滑动窗口 | 仅保留最近256个tokens |
| StreamingLLM(4,256) | Sink Cache | 保留前4个+最近252个tokens |
| Quantization (KIVI) | 量化压缩 | 2-bit量化KV值以节省空间 |
| Offloading | 缓存卸载 | 将非当前层KV卸载至CPU内存 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

#### ✅ **表1：标准常识与语言建模任务表现**
| Method | Avg. Acc | Wiki PPL | LMB PPL |
|--------|----------|----------|---------|
| Full Attention | 72.27 | 7.54 | 3.14 |
| SWA / StreamingLLM | ~72.13 | ~7.54 | ~3.15 |
| Quantization | 72.27 | 7.54 | 3.14 |
| **ARM (Ours)** | **71.89** | **7.86** | **3.21** |

> 💡 结论：在短上下文任务上，ARM性能略有下降但仍在合理范围内，说明其结构适配未破坏基本能力。

---

#### ✅ **表2：短上下文回忆密集型任务**
| Method | Avg. Score |
|--------|------------|
| Full Attention | 25.31 |
| SWA(256) | 17.41 ⬇️ |
| StreamingLLM | 25.31 |
| Quantization | 25.29 |
| **ARM (Ours)** | **25.40** ✅ |

> 💡 结论：ARM在需要精确回忆的任务上优于滑动窗口类方法，接近全注意力表现，表明**软写入有效保留了关键信息**。

---

#### ✅ **表3：LONGBENCH 长上下文任务表现**
| Method | Average Score |
|--------|----------------|
| Full Attention | ~18.0 |
| SWA(256) | ~2.88 |
| StreamingLLM | ~10.76 |
| Quantization | ~16.98 |
| **ARM (Ours)** | **~18.02** ✅ |

> 特别亮点：
> - 在 `TRC`（多跳问答）任务上达到 **62.50**，远超第二名（38.60）
> - 表明ARM能更好地支持深层推理

---

#### ✅ **表4：RULER 不同长度下的表现**
| Context Length | Full Attention | ARM (Ours) |
|----------------|---------------|-------------|
| 4K             | 14.23         | **32.67** ✅ |
| 8K             | 6.90          | **17.70** ✅ |
| 16K            | 7.85          | **10.01** ✅ |
| 32K            | 5.04          | **8.55** ✅ |
| 64K            | 5.84          | **6.65** ✅ |
| 128K           | OOM           | **5.78** ✅ |

> 💥 结论：ARM不仅在极长上下文中仍可运行，且在所有长度下均显著优于其他方法，**唯一能在128K成功运行的方法**。

---

### **消融实验结果（Ablation Study）**
见 **Table 9**，分析两个核心设计的影响：

| Variant | Gated Write | Learnable Sparsity | Avg. Score |
|--------|-------------|--------------------|------------|
| FIFO(256) | × | × | 17.41 |
| Fixed Sparsity (1/2) | ✓ | × | 24.42 |
| Fixed Sparsity (1/4) | ✓ | × | 24.94 |
| **ARM (Full)** | ✓ | ✓ | **25.40** ✅ |

> 🔍 发现：
> - **Gated Write 贡献最大**：从17.41 → 24.94，说明软融合对信息保留至关重要。
> - **Learnable Sparsity 进一步提升**：允许模型根据输入动态调节检索范围，提升灵活性与效率。

---

## **4. 关键结论和发现**

### **主要发现**
1. **软淘汰优于硬淘汰**：
   - Gated memory update 能在有限空间内更有效地混合信息，减少关键信息丢失。
   - 理论分析表明该策略在不确定性下是Bayes-optimal的。

2. **动态稀疏性优于静态稀疏性**：
   - 自适应Top-M检索使模型能“智能地”分配计算资源，简单任务快，复杂任务准。

3. **ARM实现了真正的可扩展长上下文推理**：
   - 在128K context下仍能运行并取得有意义结果，而多数基线已OOM或性能崩溃。
   - 同时保持较低延迟和内存占用。

4. **效率优势明显**：
   - 图2显示：ARM的**解码延迟显著低于offloading方案**（后者因CPU-GPU传输慢10倍以上）。
   - 内存使用与sliding window相当，远低于full attention。

---

### **方法的局限性**
- **预训练依赖**：需额外预训练Gumbel-write和MDP-read模块，不能直接插拔到已有模型。
- **路由精度限制**：若路由器未能正确聚合同类token，可能导致信息干扰。
- **超参数敏感**：memory slot数量、树深度等影响性能平衡。
- **理论最优性假设强**：信息分配最优性的证明基于i.i.d. Gaussian等理想假设。

---

### **未来工作方向**
1. **无限上下文（infinite context）探索**：
   - 结合ARM与循环机制（recurrence）或状态空间模型（如Mamba），追求真正无界记忆。

2. **亚线性内存增长设计**：
   - 探索memory size随$\sqrt{T}$或$\log T$增长的可能性，进一步提升可扩展性。

3. **跨层共享与异构路由**：
   - 不同Transformer层使用不同路由策略，适应层次化语义。

4. **硬件协同优化**：
   - 设计专用kernel支持Gumbel-Softmax与parallel scan风格的gate update，加速prefill阶段。

---

> 🧠 **总体评价**：  
> ARM 提出了一种**结构性创新**而非工程修补，将KV-Cache重新构想为一个**可学习、可路由、可控制的神经记忆系统**。它统一了sparse attention与dynamic memory network的思想，在性能、效率、可扩展性之间取得了卓越平衡，为下一代高效长上下文LLM提供了重要范式。

</details>

---

### 4. [RBS-Attention: Radius-Bounded Sparse Prefill for Long-Context Large Language Models](https://arxiv.org/abs/2609.20971)

**Authors**: Chuxu Song, Jiuqi Wei, Zhencan Peng  
**Category**: cs.AI  
**Published**: 2026-09-22  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.20971v1  

#### Abstract
Long-context large language model inference is increasingly limited by prefill, where dense self-attention processes the entire prompt before generation begins. Sparse block selection can reduce this cost, but a block centroid may hide a highly relevant token among many irrelevant ones. We call this...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：RBS-Attention: Radius-Bounded Sparse Prefill for Long-Context Large Language Models**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
在长上下文大语言模型（LLM）推理中，**prefill 阶段的自注意力计算是主要延迟瓶颈**。传统的 dense self-attention 在处理数十万 token 上下文时面临 $O(N^2)$ 的计算复杂度，导致时间到首词（Time-to-First-Token, TTFT）显著增加。

现有稀疏化方法（如 FlashPrefill、XAttn）通过选择关键的 key-value 块来减少计算量，但存在一个关键缺陷：  
> **Mean Dilution（均值稀释）问题**：当一个关键块（key block）中包含少量高度相关 token 和大量无关 token 时，其**块质心（centroid）** 的平均对齐得分可能很低，导致整个块被错误丢弃，从而丢失重要信息。

### **提出了什么新方法或新思路**
作者提出 **RBS-Attention**（Radius-Bounded Sparse Attention），一种无需训练的稀疏 prefill 方法，核心思想是：

- 引入 **双分支选择机制（dual-branch selection）**：
  - **Base Branch（基础分支）**：基于块质心（centroid）的 relevance 得分，保留平均对齐度高的块。
  - **Rescue Branch（救援分支）**：引入 **最大键半径（maximum key-block radius）** 作为风险信号，识别那些内部差异大、易发生 mean dilution 的块，并给予“救援”机会。
    - 半径 $r_b = \max_{k \in K_b} \|k - c_b\|_2$ 表示块内 token 到质心的最大距离。
    - 使用当前 prompt、层、头的半径分布动态计算一个缩放系数 $\beta_b$，实现 **radius-adaptive** 调整。

- **独立阈值 + 掩码联合（independent thresholds & mask union）**：
  - 两个分支分别应用相对阈值（relative threshold），生成各自的候选块集合。
  - 最终选择集为两者的并集，确保高相关性和高风险块都能被保留。

### **相比现有方法的优势**
- **更鲁棒的选择机制**：避免因 mean dilution 导致的关键信息丢失。
- **无需训练**：完全基于运行时统计，部署简单。
- **保持高效执行**：仍使用 block-sparse FlashAttention 内核，保证 GPU 利用率。
- **内容自适应密度**：实际保留密度随输入内容变化，而非固定比例。

---

## 2. **核心实验方法和设置**

### **使用的数据集**
- **RULER**：评估长上下文检索与推理能力，涵盖不同长度任务。
- **LongBench-v2**：多任务长文本理解基准，包含短、中、长三种长度任务。
- **InfiniteBench**：扩展至超过 100K token 的长上下文评测。
- **Video-MME**：多模态视频理解基准，用于评估视觉-语言模型表现。

### **实验设置和评估指标**
- **模型**：
  - 主要模型：`Qwen3-30B-A3B-Instruct-2507-FP8`（MoE 架构）
  - 对比模型：`Qwen3-32B`（dense）、`Qwen3-VL-30B-A3B-Thinking-FP8`（multimodal）
- **硬件**：
  - 主要系统性能测试：**H100 GPU**
  - 补充实验：A100 GPU（TP=2 或 TP=4）
- **上下文长度**：16K ~ 256K，重点关注 128K 场景。
- **评估指标**：
  - **速度指标**：
    - Standalone prefill-attention speedup
    - vLLM prefill-attention speedup
    - End-to-end TTFT speedup
  - **质量指标**：
    - RULER 准确率（%）
    - LongBench-v2 总体得分（unit interval）
    - Video-MME 总分（%）
  - **效率指标**：
    - 实际密度（actual density）
    - 峰值内存占用（peak memory）

### **基线方法对比**
- **Sparse Prefill Baselines**：
  - FlashPrefill：基于质心 + 相对阈值
  - FlexPrefill：上下文感知稀疏注意力
  - MInference：动态稀疏注意力
  - XAttn：反向对角线评分块稀疏
- **控制变量对比**：
  - Centroid-only：仅使用质心得分
  - Full-L2：使用完整 L2 上界（未衰减）
  - Quest-style：坐标级 min/max 边界选择（类似 Quest）

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**
在 `Qwen3-30B-A3B-Instruct-2507-FP8` 模型上，**128K 上下文长度**下的核心加速结果：

| 指标 | RBS-Attention | 加速倍数 |
|------|----------------|----------|
| Standalone prefill-attention speedup | — | **20.65×** |
| vLLM prefill-attention speedup | — | **11.92×** |
| End-to-end TTFT speedup | — | **5.97×** |

> 这是目前公开文献中在该规模模型上的最高速度提升之一。

### **与基线方法的质量对比**
#### **RULER-128K 准确率（%）**
| 方法 | 准确率 |
|------|--------|
| Dense | 89.69 |
| FlashPrefill | 85.01 |
| XAttn | 71.01 |
| MInference | 87.18 |
| **RBS-Attention** | **88.36** |

> RBS 在仅保留约 11% 块的情况下达到接近 dense 的准确率，显著优于其他稀疏方法。

#### **LongBench-v2 总体得分**
| 模型 | 方法 | 总分 | 估计密度 |
|------|------|------|---------|
| Qwen3-32B | Dense | 0.394 | 100% |
| Qwen3-32B | RBS-Attention | **0.376** | **9.719%** |
| Qwen3-30B-A3B | Dense | 0.388 | 100% |
| Qwen3-30B-A3B | RBS-Attention | **0.370** | **7.912%** |

> RBS 在更低密度下实现了与最优稀疏方法相当甚至更好的性能。

### **消融实验结果**
#### **控制实际密度下的选择器比较（5.34% 密度）**
| Selector | Accuracy |
|---------|----------|
| Centroid-only | 73.63 |
| Full-L2 bound | 72.41 |
| Quest-style | 73.71 |
| **RBS adaptive union** | **76.67** |

> 表明：
> - 完全依赖 L2 上界并不优于简单的质心得分；
> - RBS 的 **adaptive rescue + dual-branch union** 设计显著优于单一策略。

#### **半径五分位分析（Figure 1c）**
- 最高半径五分位（Q5）包含了 **39.5% 的 top-5% attention blocks**（均匀分布预期为 20%）。
- 说明高分散块中确实富含关键信息，验证了“救援”机制的必要性。

---

## 4. **关键结论和发现**

### **主要发现**
1. **Mean Dilution 是真实且严重的问题**：在长上下文场景中，关键信息常隐藏于异质性强的块中，传统基于质心的方法容易误删。
2. **Block Radius 是有效的风险信号**：最大半径能有效标识出可能发生 mean dilution 的块。
3. **Adaptive Rescue 优于保守边界**：并非所有高半径块都应无条件保留；通过分布自适应的 $\beta_b$ 调节救援强度，比直接使用 Full-L2 上界更有效。
4. **Dual-Branch Union 提升鲁棒性**：独立阈值 + 并集操作，既保留了高效路径，又增加了容错能力。
5. **高性能与高质量可兼得**：RBS 在实现近 **21× prefill 加速**的同时，仅损失不到 1% 的 RULER 准确率。

### **方法的局限性**
- **主要适用于长上下文 prefill**：在短序列中选择开销可能抵消收益。
- **Decode 阶段未优化**：当前设计聚焦 prefill，decode 仍需标准 KV-cache。
- **阈值依赖调优**：虽然有自适应机制，但 $\alpha_{base}, \alpha_{rescue}$ 仍需离线校准。
- **临时工作空间较大**：双分支计算带来更高中间内存占用（见 Table 15）。

### **未来工作方向**
- 将 radius-adaptive 思想扩展至 **decode-time KV 缓存检索**。
- 探索更轻量化的 radius 统计方式以降低 selector 开销。
- 结合 pattern-based 方法（如 local window, retrieval column）进一步提升选择精度。
- 研究如何将此类机制集成进端到端训练框架中，实现 joint optimization。

---

> ✅ **总结一句话**：  
> RBS-Attention 通过引入 **radius-adaptive dual-branch selection**，有效缓解了长上下文 prefill 中的 mean dilution 问题，在几乎不损失模型性能的前提下，实现了高达 **20.65× 的 prefill 加速**，为超长上下文 LLM 推理提供了高效可靠的解决方案。

</details>

---

### 5. [Towards Full Pipeline FP8 Reinforcement Learning for LLMs](https://arxiv.org/abs/2609.22870)

**Authors**: Fanchao Chen, Ziheng Jiang, Ziyun Wei, Zheng Zhong, Du Li, Chi Zhang, Haibin Lin, Shivaram Venkataraman  
**Category**: cs.LG  
**Published**: 2026-09-22  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.22870v1  

#### Abstract
Reinforcement learning (RL) has become a key technique for improving the reasoning and agentic abilities of large language models (LLMs). Although FP8 quantization can accelerate RL training, maintaining stability throughout an FP8 RL pipeline remains challenging. While previous works have focused o...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Towards Full Pipeline FP8 Reinforcement Learning for LLMs

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

本文揭示并解决了**全流水线 FP8 强化学习（Full-Pipeline FP8 RL）训练中的严重不稳定性问题**。尽管 FP8 已被成功用于 LLM 的预训练和推理，但在 RL 训练中，即使采用统一的 FP8 流水线（rollout 和 training 都用 FP8），仍会出现**中期熵激增（mid-training entropy surges）** 和输出混乱（garbled outputs）的现象。

作者指出，这一现象的根本原因并非此前研究关注的“train-rollout mismatch”，而是被忽视的 **compounded FP8 quantization noise（累积的 FP8 量化噪声）对 PPO 风格算法中 clipping 机制的破坏**：

- 量化噪声在计算 importance ratio $ r(\theta) = \pi_\theta / \pi_{\text{old}} $ 时被放大；
- 特别是对于负优势（negative-advantage）token，其重要性比率被错误地推至下界 clipping 阈值以下（$ r^{FP8} \leq 1-\epsilon $），导致梯度被提前归零；
- 结果：模型无法有效惩罚病态输出（如乱码），这些输出不断积累，最终引发训练崩溃。

---

### 🛠️ 提出了什么新方法或新思路

提出 **Calibrated Clipping（校准剪裁）** ——一种动态调整 FP8 训练中 clipping 边界的轻量级方法，旨在恢复高精度（BF16）下的信任域语义。

该方法分为两个阶段：

1. **Lower-Bound Alignment（下界对齐）**  
   利用 BF16 shadow pass 获取高精度的重要性比率分布，确定在负优势 token 中应被 clipping 的分位数 $ \alpha_{\text{low}} = F_{BF16}^-(1-\epsilon) $，然后在 FP8 分布中找到对应此分位数的值作为新的下界 $ L $，确保两种精度下被 clipping 的 token 数量一致。

2. **Upper-Bound Rebalancing（上界重平衡）**  
   为防止因下界放宽而导致过度惩罚，通过调节上界 $ H $ 来保持正负更新项的比例 $ p = \frac{\text{positive contribution}}{\text{negative contribution}} $ 与 BF16 基线一致。

此外，采用**周期性重校准策略**（每 20 步一次），避免频繁执行 BF16 推理带来的开销，实现效率与稳定性的平衡。

---

### ⚖️ 相比现有方法的优势

| 方法 | 是否解决 train-rollout mismatch | 是否解决 FP8 clipping distortion | 是否支持 full-pipeline FP8 |
|------|-------------------------------|-------------------------------|--------------------------|
| TIS / MIS | ✅ | ❌ | ❌（混合精度） |
| Unified FP8 / Jet-RL | ✅ | ❌ | ✅ |
| **Calibrated Clipping (本文)** | ✅（继承 Unified FP8） | ✅ | ✅ |

- **首次识别出 FP8 量化噪声对 clipping 机制的系统性扭曲问题**；
- **无需修改网络结构或损失函数**，仅需微调 clipping 参数；
- **几乎无额外计算成本**（周期性执行），却能显著提升稳定性；
- 在多个算法（GRPO、DAPO）、模型规模（8B~32B）、scaling granularity（tensorwise/rowwise/blockwise）下均有效。

---

## 2. 核心实验方法和设置

### 📚 数据集

- **DeepScaleR**：用于 GRPO 实验，包含长上下文推理任务（context length 达 16K）；
- **DAPO-Math-17K**：来自原始 DAPO 论文的数据集，专注于数学推理；
- **Eurus coding dataset**（附录）：用于代码生成任务评估。

---

### 🔧 实验设置

| 组件 | 设置 |
|------|------|
| **模型** | Qwen3-8B-Base, Qwen3-14B-Base, Qwen2.5-32B |
| **RL 算法** | GRPO（无 critic）、DAPO（带非对称 clipping） |
| **FP8 scaling granularities** | Tensorwise、Rowwise、Blockwise |
| **Rollout & Training Backend** | vLLM（inference）、TorchAO（FP8 training）、VeRL（RL 框架） |
| **TIS 使用** | 所有实验默认启用，截断阈值 $ C=2 $ |
| **Recalibration Frequency** | 每 20 步进行一次 BF16 shadow pass 进行参数校准 |
| **优化器** | AdamW，学习率 $ 1e^{-6} $，batch size 256 |

---

### 🎯 评估指标

- **主性能指标**：在多个推理基准上的平均准确率：
  - AIME24/25（Avg@16 或 Avg@32）
  - AMC23/24
  - MATH-500
  - Gaokao
  - Minerva Math
  - OlympiadBench
  - TACO / APPS / Codeforces（编码任务）
- **训练稳定性指标**：
  - Actor entropy（检测是否出现 surge）
  - Reward 曲线
  - Response length
  - Clipping fraction
  - PPO surrogate objective 分解（正负更新比例）

---

### 🆚 基线方法对比

| 基线配置 | 描述 |
|---------|------|
| `BF16` | 全流程 BF16 精度，视为性能上限 |
| `FP8-Rollout + BF16-Train` | 混合精度，存在 train-inference mismatch |
| `Unified FP8 (vanilla)` | 全流程 FP8，但未做任何 clipping 校正 |
| `Unified FP8 + Calibrated Clipping` | 本文方法 |
| （附录）CISPO、BAPO | 其他 clipping 改进方法，用于消融分析 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Table 1 & 2）

#### ✅ GRPO 实验（Qwen3-8B-Base）

| 方法 | 平均得分 | 相比 BF16 下降 |
|------|--------|-------------|
| BF16 | 57.6 | — |
| Vanilla Tensorwise FP8 | 46.1 | ↓11.5 |
| Vanilla Rowwise FP8 | 47.0 | ↓10.6 |
| Vanilla Blockwise FP8 | 54.1 | ↓3.5 |
| **+ Calibrated Clipping** | **58.6** | ↑**+1.0** |

👉 **结论**：Calibrated Clipping 不仅完全恢复性能，甚至略微超越 BF16 基线（尤其在 blockwise 设置下）。

#### ✅ DAPO 实验（Qwen3-14B-Base）

| 方法 | AIME24 Avg@32 |
|------|--------------|
| BF16 | 50.9 |
| Vanilla Blockwise FP8 | 41.6 |
| **+ Calibrated Clipping** | **47.4** |
| 提升幅度 | **+5.8 pts** |

👉 即使在更高上界（鼓励探索）的 DAPO 中，FP8 也会导致更早崩溃，而本方法可成功复现 BF16 的熵演化轨迹。

---

### 🔍 消融实验结果（附录 A.5）

测试了不同超参数敏感性（Qwen3-8B rowwise FP8 setting）：

| 设置 | 平均得分 |
|------|--------|
| 默认（每 20 步校准） | 56.51 |
| 每 10 步校准 | 58.31 |
| 每 40 步校准 | 57.91 |
| 初始化 [0.6, 1.8] | 56.51 |
| 初始化 [0.8, 1.2] | 56.20 |

✅ **发现**：所有变体都显著优于 vanilla FP8（47.0），说明方法具有较强鲁棒性。

---

### ⚙️ 性能增益（Throughput）

| Scaling Granularity | 相对于 BF16 的训练吞吐提升 |
|--------------------|---------------------------|
| Tensorwise FP8 | **up to 1.5×** |
| Blockwise FP8 | ~1.1–1.2× |

结合 FP8 rollout 可带来约 30% 的生成加速，**整体 pipeline 效率大幅提升**。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **FP8 RL 的根本挑战不仅是 train-rollout mismatch，更是 quantization-induced clipping distortion**：
   - 量化误差在 importance ratio 上被放大；
   - 导致负优势 token 被“误剪裁”（over-clipped），梯度消失，无法抑制垃圾输出。

2. **熵激增的本质是“惩罚失效”而非“探索失控”**：
   - 高熵样本多为乱码，且多数具有负优势；
   - 应被抑制却因 FP8 噪声逃逸惩罚机制。

3. **Calibrated Clipping 能精准修复信任域边界**：
   - 动态对齐 clipping quantile 和 update ratio；
   - 成功消除 entropy surge，恢复 BF16 级性能。

4. **方法通用性强**：
   - 在 GRPO、DAPO、8B~32B 模型、多种 scaling granularity 下均有效；
   - 可无缝集成到现有 FP8 RL 流水线中。

---

### ⚠️ 方法的局限性

- **依赖周期性 BF16 shadow pass**：虽然频率低（每 20 步），但仍引入额外计算负担，不适合极端低延迟场景；
- **假设旧策略 $\pi_{\text{old}}$ 和当前策略 $\pi_\theta$ 的量化行为相似**：若两者量化误差差异大，可能影响校准效果；
- **目前仅适用于 token-level clipping**：对于 sequence-level clipping（如 GSPO），需进一步扩展；
- **未解决其他潜在的 FP8 敏感模块**（如 critic loss、KL penalty）。

---

### 🔮 未来工作方向

1. **将 Calibrated Clipping 扩展至 sequence-level 或 group-level policy optimization**；
2. **探索免 BF16 参考的自适应校准机制**（例如基于 FP8 内部统计估计）；
3. **研究其他 FP8 敏感组件的稳定性问题**（如 value function 更新）；
4. **应用于 MoE 架构或更大规模模型的 FP8 RL 训练**；
5. **结合 sparse update 或 gradient compression 技术进一步提升端到端效率**。

---

## ✅ 总结

> **Calibrated Clipping 是迈向高效、稳定、全流水线 FP8 RL 的关键一步**。它以极小代价解决了长期被忽视的“量化噪声破坏信任域”问题，使得 FP8 不再只是推理加速工具，而真正成为 LLM 强化学习训练的可行选择。

</details>

---

### 6. [VISTA: An Attention-Based Multi-Agent Reinforcement Learning Architecture for Space Situational Awareness Sensor Tasking](https://arxiv.org/abs/2609.23875)

**Authors**: Miguel Leiva-V\'elez, Adalberto Claudio Quiros, Nicolas Gaston Rozado, Hodei Urrutxua, V\'ictor Rodr\'iguez-Fern\'andez  
**Category**: cs.LG  
**Published**: 2026-09-22  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.23875v1  

#### Abstract
The rapid growth of resident space objects is increasing the complexity of space situational awareness sensor tasking, challenging classical optimization methods as they allocate finite, heterogeneous, and distributed sensing resources across ever-larger catalogues. Existing deep reinforcement learn...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《VISTA: An Attention-Based Multi-Agent Reinforcement Learning Architecture for Space Situational Awareness Sensor Tasking》总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
随着在轨空间物体（Resident Space Objects, RSOs）数量迅速增长，**空间态势感知**（Space Situational Awareness, SSA）中的传感器任务分配（sensor tasking）面临巨大挑战。传统优化方法难以高效处理大规模、动态变化的目标目录和分布式异构传感器网络。现有的深度强化学习（DRL）方法受限于**固定维度的状态和动作表示**，无法灵活适应目标数量、传感器配置的变化，导致可扩展性和泛化能力不足。

### 提出的新方法与创新思路
本文提出了 **VISTA**（Variable-Entity Intelligent Sensor Tasking Architecture），一种基于注意力机制的可扩展多智能体强化学习架构，用于解决上述问题。其核心创新包括：

- **实体中心化表示**（Entity-centric representation）：将传感器和空间目标建模为“实体”（entity），每个实体独立编码，避免固定维度输入/输出槽位。
- **Top-K 检索机制**：结合物理规律和任务需求，从完整目录中筛选出最多 K 个决策相关的候选目标，限制上下文长度，实现计算成本与性能的权衡。
- **注意力机制**（Attention）：通过多头自注意力（MHSA）建模传感器与候选目标之间的几何、信息和协作关系，实现对动态集合的关系推理。
- **指针网络解码器**（Pointer-based action decoding）：使用指针网络选择当前候选集中的目标进行观测，动作空间随候选集动态变化，而非固定标签。
- **循环记忆集成**（LSTM）：保留历史任务决策信息，支持长期规划。
- **参数共享的去中心化多智能体框架**：所有传感器共享策略参数，但拥有独立的循环状态，支持异构传感器协同。

### 相比现有方法的优势
- **可扩展性强**：模型大小与目标目录大小、传感器数量无关，适用于从小规模到数万目标的大规模场景。
- **泛化能力强**：可在未训练过的目录大小、轨道分布、初始不确定性等条件下实现零样本迁移（zero-shot generalization）。
- **高效且鲁棒**：相比传统方法和固定维度 DRL 基线，在恢复速度、不确定性降低等方面显著领先。
- **支持异构传感器协同**：能根据不同传感器的能力（如探测距离响应函数）自适应分配任务。

---

## 2. 核心实验方法和设置

### 数据集与仿真环境
- 使用作者自研的 **C++ 仿真器**，集成 PufferLib 实现高吞吐并行训练。
- 仿真器模拟轨道传播、协方差演化、传感器可见性、指向动力学、伪测量更新等过程，但抽象了检测、数据关联和精密定轨链路。
- 所有实验均在相同接口下运行，确保公平比较。

### 实验设置
共设计三个阶段实验，覆盖不同复杂度场景：

| 阶段 | 场景描述 | 传感器数 | RSO 数量 | Top-K (K) |
|------|--------|---------|----------|----------|
| Phase I | 固定单传感器基准测试 | 1 | 30 | 30 |
| Phase II | 大规模 LEO-to-LEO 协同调度 | 12 | 2000 → 20,000 | 60 |
| Phase III | 异构传感器协同任务分配 | 4（2种类型） | 240 | 60 |

### 评估指标
- **平均目录不确定性**（mean catalogue uncertainty）：所有 RSO 的径向、切向、法向位置标准差之和的均值。
- **恢复时间**（Recovery time）：平均不确定性首次降至 1 km 以下的时间。
- **终端不确定性**（Terminal uncertainty）：5.56 小时后的平均不确定性。
- **有效覆盖率**（Effective coverage）：衡量观测在目标间的分布均匀性。
- **非重复率**（Non-duplication rate）：避免多个传感器同时观测同一目标的比例。
- **零样本迁移性能**：使用难度调整的退化度量 $ D_m $ 衡量模型在分布外条件下的鲁棒性。

### 基线方法对比
- **非学习基线**：
  - Random：随机可行选择
  - Oldest-first：优先观测最久未观测目标
  - Max-U：优先观测不确定性最高的目标
  - EIG（Expected Information Gain）：基于一步信息增益预测
  - Beam Search：轻量级回溯搜索（beam width=4, depth=3）
- **学习基线**：
  - **Flat LSTM**：固定维度的 MLP-LSTM 架构，使用分类输出头，是早期 SSA DRL 研究常用结构。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### Phase I：小规模固定场景（1传感器，30 RSO）
- VISTA 达到 1 km 平均不确定性的**恢复时间比 Flat LSTM 快 31.2%**（13,245s vs 19,245s）。
- 终端不确定性为 0.427 km，比 EIG 低 44.6%，比 Flat LSTM 显著更低。
- 在零样本迁移测试中（如目录扩大至 120 RSO），VISTA 终端不确定性比 Flat LSTM 低 **95.0%**，且退化度 $ D_m = 0.211 $，远优于 LSTM 的 2.398，显示更强泛化能力。

#### Phase II：大规模协同调度（12传感器，2000 RSO）
- 5小时后，VISTA 平均不确定性仅为 **0.0846 km**。
- 相比最强经典方法 EIG，不确定性降低 **97.5%**。
- 相比 Flat LSTM，不确定性降低 **99.3%**。
- 在零样本扩展至 **20,000 RSO** 的测试中，仍保持近线性扩展特性，恢复时间与 $ \frac{N_{\text{RSO}}}{N_{\text{sensors}}} $ 呈强相关（$ R^2 = 0.882 $）。
- 成功网络最小配置显示：约每 **412 个 RSO 需 1 个传感器**即可完成恢复任务。

#### Phase III：异构传感器协同（4传感器，240 RSO）
- 不同类型传感器表现出明显能力适配行为：
  - 近程型（Type A）平均观测距离 ~1550–1640 km
  - 远程型（Type B）平均观测距离 ~2310–2400 km
  - 与预设响应函数（交叉点 1900 km）高度一致。
- **平均非重复率达 98.27%**，所有 240 个目标在每轮实验中均被至少观测一次。
- **有效覆盖率达 85.76%**，表明任务分配广泛且均衡。
- 模态嵌入消融实验证明：若交换或中和传感器类型标签，性能分别下降 75.9% 和 13.5%，说明模型确实利用了传感器模态信息。

### 消融实验与分析
- **Top-K 上下文敏感性**：当 $ K $ 从 15 增加到 60，终端不确定性下降 86.7%；继续增加至 90 改进有限，表明存在收益递减点。
- **注意力特征关联分析**：注意力权重与目标不确定性、观测年龄正相关，与指向需求（slew）负相关，符合“优先处理高不确定性且易到达目标”的策略。
- **表征可视化**（UMAP）：编码器和循环状态在不同目录规模下沿相似流形演化，但在轨道倾角剧变时出现分支，解释了特定迁移失败的原因。

---

## 4. 关键结论和发现

### 主要发现
1. **VISTA 显著优于现有方法**：在多种场景下，无论是恢复速度还是最终不确定性控制，VISTA 均大幅超越经典启发式方法和主流 DRL 基线。
2. **具备强大零样本泛化能力**：无需重新训练即可适应从未见过的目录规模、初始状态和轨道构型，尤其在目录扩展上表现稳健。
3. **支持异构传感器自适应协同**：能够学习不同类型传感器的能力差异，并据此进行互补性任务分配。
4. **近线性可扩展性**：系统性能与 $ \frac{\text{RSO 数量}}{\text{传感器数量}} $ 呈近似线性关系，为实际部署提供容量估算依据。
5. **内部表征具有物理意义**：注意力机制关注的任务特征（如不确定性、年龄、slew）与直觉一致，且循环状态能反映目录恢复进程。

### 方法的局限性
- 当前仿真器简化了真实物理效应（如摄动、机动、不完美通信），尚未在高保真 SSA 链中验证。
- 轨道几何多样性不足（如逆行轨道迁移失败），训练分布影响零样本性能。
- 未考虑通信延迟、带宽限制和异步执行，去中心化假设较强。
- Top-K 检索依赖手工设计的排序规则（如不确定性+年龄），未来可探索端到端学习。

### 未来工作方向
- 将 VISTA 集成到更完整的 SSA 链中，包含真实检测、数据关联和轨道确定模块。
- 探索更复杂的轨道动力学和传感器模型（如雷达 RCS 变化、光学信噪比）。
- 研究完全去中心化、异步更新和有限通信下的多智能体协调。
- 对 VISTA 各组件进行受控消融实验，明确注意力、指针、Top-K 等模块的具体贡献。
- 扩展至地月空间（cislunar）等更复杂几何环境下的任务调度。

--- 

> **代码与复现**：项目代码和冻结实验配置已开源：https://github.com/RocketNeurons/VISTA-SSA

</details>

---

### 7. [TreeSpark: Calibrated, Load-Adaptive Draft Trees for Semi-Autoregressive Speculative Decoding](https://arxiv.org/abs/2609.22098)

**Authors**: Huapeng Zhou, Huayu Wang, Xinyu Wang  
**Category**: cs.CL  
**Published**: 2026-09-22  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.22098v1  

#### Abstract
Speculative decoding accelerates language-model inference by letting a cheap drafter propose tokens that the target model verifies in parallel. Recent block drafters make drafting nearly free: a single backbone pass emits an entire block of draft tokens. Draft trees promise a further gain -- several...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：TreeSpark: Calibrated, Load-Adaptive Draft Trees for Semi-Autoregressive Speculative Decoding**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现有的 **draft tree** 方法在结合 **semi-autoregressive drafters**（如 DSpark）时面临两个核心挑战：
1. **候选节点排序不合理**：现有方法（如 DDTree）基于每个位置的边际分布（marginal distribution）构建树，忽略了父节点对子节点条件概率的影响，导致生成大量低质量、不连贯的分支。
2. **固定树大小缺乏适应性**：大多数方法采用固定大小的 draft tree，无法根据当前解码轮次的不确定性或系统负载动态调整验证开销，在高并发场景下可能降低吞吐量。

此外，在非零温度采样中，若使用确定性 top-k 扩展并进行递归拒绝采样（recursive rejection sampling），会导致输出分布偏移，破坏解码的无损性（lossless）。

---

### **提出的新方法与创新思路**
TreeSpark 提出了一套完整的、适用于 semi-autoregressive drafters 的 **条件化、自适应 draft tree 构造框架**，其核心创新包括：

#### ✅ **(1) 利用 Markov Head 实现 parent-conditioned 条件分布**
- 复用 DSpark 中已有的低秩 **Markov head** $ B(x_p, \cdot) $，为每个节点提供以父 token 为条件的概率分布：
  $$
  q_o(\cdot|x_p) = \text{softmax}(U_d + B(x_p, \cdot))
  $$
- 在单次 backbone 推理后，仅通过 embedding 查找和矩阵乘法即可获得不同 parent 下的子代分布，无需额外前向传播。
- **优势**：避免了“同一深度所有分支共享相同子分布”的错误假设，显著提升路径合理性。

#### ✅ **(2) 边接受率校准（Edge Acceptance Calibration）**
- 使用目标模型的验证标签训练一个两参数 Platt 校准器，将原始条件概率映射为更准确的边接受估计：
  $$
  p(\text{edge}) = \sigma(a \cdot \text{logit}(q_o) + b)
  $$
- 关键是**拟合群体限定为祖先已被接受的边**（survival-conditioned population），确保估计值可用于路径生存概率建模。
- **优势**：相比未校准的 raw $ q_o $ 或 confidence head，校准后的预测 ECE 降至 0.0105，AUC 达 0.948。

#### ✅ **(3) 基于路径生存的自适应停止机制**
- 使用路径上各边接受概率的乘积作为节点优先级（best-first expansion）。
- 动态停止条件：当最优剩余候选的路径生存概率低于阈值 $ \theta $ 时终止扩展。
- $ \theta $ 可解释为“每增加一个验证 token 所需的最小期望收益”，成为运行时控制变量。

#### ✅ **(4) 无损采样策略：无放回抽样 + 递归拒绝采样**
- 子节点从残差分布中**无放回抽样**（without replacement）生成，防止重复。
- 验证阶段根据 backbone 输出和抽样顺序重建 proposal 分布，执行真正的递归拒绝采样。
- **理论保证**：任意温度下均保持目标分布不变（Proposition 1）。

#### ✅ **(5) 负载感知的运行时调度接口**
- 引入“价格阶梯”（price ladder）机制，由服务调度器根据 batch size 和温度选择 $ \theta $。
- 高负载时自动缩小树规模甚至退化为 chain，实现 graceful degradation。
- 实现了**树形状作为可控的服务变量**。

---

### **相比现有方法的优势**
| 维度 | TreeSpark | 其他方法（如 DDTree, PCTree） |
|------|-----------|-------------------------------|
| 条件建模 | ✅ 显式 parent-conditioned | ❌ 多数使用 marginal 或近似 conditioning |
| 自适应性 | ✅ 动态 per-round 停止 + 负载适配 | ❌ 固定节点预算 |
| 采样正确性 | ✅ 严格 lossless at any T | ❌ 常见 deterministic 扩展引入偏差 |
| 实现代价 | ✅ 无需重训练，复用现有 drafter | ✅ 同类方法也多为 plug-in 设计 |
| 性能表现 | ⬆️ 显著优于 chain 和 marginal tree | ➖ 多数仅报告贪婪解码 |

---

## **2. 核心实验方法和设置**

### **数据集**
- 主要评测任务（共6项）：
  - **GSM8K**, **MATH-500**, **HumanEval**, **MBPP**, **MT-Bench**, **Alpaca**
- 补充任务（新增3项）：
  - **AIME25**, **LiveCodeBench**, **Arena-Hard-v2**
- Prompt 数量：
  - 贪婪解码：150 prompts（每任务25个）
  - 采样与自适应评估：17–25 个 calibration-disjoint prompts 每任务

### **模型设置**
- **Target Models**：Qwen3-4B / 8B / 14B
- **Drafters**：
  - `dspark_qwen3_*-block7`：semi-autoregressive drafter，含 Markov head 与 confidence head
  - `dflash_qwen3_*-block7`：marginal drafter（用于对比）

### **评估指标**
| 指标 | 定义 | 说明 |
|------|------|------|
| **T** | 平均每轮 target forward 接受的 draft token 数 | 核心效率指标，越高越好 |
| **AR Speedup** | speculative decoding 相对于纯 target-only decoding 的 wall-clock 加速比 | 实际推理速度体现 |
| **Verified Nodes** | 每轮参与验证的 draft token 总数 | 衡量验证开销 |
| **Utilization** | 成功提交路径上的节点占比 | 反映树的有效性 |
| **Goodput** | 单位时间生成的有效 token 数 | 服务场景下的综合性能 |
| **Total Variation (TV)** | 输出分布与真实目标分布的距离 | 评估采样无损性 |

### **基线方法对比**
| 方法 | 类型 | 描述 |
|------|------|------|
| **DFlash Chain** | Chain | 基于 marginal drafter 的链式推测 |
| **DDTree** | Fixed-budget Tree | 在 DFlash 上构建 best-first marginal tree |
| **DSpark Chain** | Chain | 当前 SOTA semi-autoregressive 链式推测 |
| **Marginal Tree on DSpark** | Ablation | 将 DDTree 应用于 DSpark 的 marginal 分布（失败案例） |
| **TreeSpark (Fixed N)** | Variant | 固定节点预算版本 |
| **TreeSpark (Adaptive θ)** | Full Method | 自适应停止 + 负载调度 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

#### 🔹 **贪婪解码下 T 提升显著**
| Model | Method | N=7 | N=14 | N=28 | N=56 |
|-------|--------|-----|------|------|------|
| Qwen3-4B | DSpark Chain | 4.39 | – | – | – |
|          | TreeSpark | **4.57** | **5.18** | **5.50** | **5.81** |
|          | ↑ 提升 | +4.1% | +18.0% | +25.3% | +32.3% |

> 在相同验证预算下，TreeSpark 比 DDTree 多接受 **9–16%** 的 token。

#### 🔹 **自适应策略全面超越固定预算**
- 在 Qwen3-4B 上（greedy, 102 prompts）：
  - 相比 $ N=28 $：**+0.073 T**，同时少验证 **5.2 tokens/round**
  - 相比 $ N=56 $：接近最大 T（差仅 0.12），但验证成本降低 **59%**

#### 🔹 **端到端加速比（AR Speedup）**
| Decoder | T=0 | T=0.5 | T=1.0 |
|--------|-----|-------|-------|
| DSpark Chain | 3.30× | 3.14× | 3.06× |
| TreeSpark (Adaptive) | **3.75×** (+14%) | **3.38×** (+8%) | **3.38×** (+10%) |

> 单请求延迟下，**解码速度快 8–14%**。

#### 🔹 **负载变化下的鲁棒性**
- 在动态批处理引擎中（A100, T=0）：
  - Chain 提升吞吐 +16% vs no speculation
  - Fixed $ N=56 $：变为负优化（224 vs 281 tok/s）
  - **Adaptive TreeSpark**：进一步提升 **+2.1%** vs chain
- 在高负载（bs=16）时，自适应控制器自动切换至 chain，避免性能下降。

#### 🔹 **采样下的无损性验证**
| 方法 | Total Variation (TV) |
|------|------------------------|
| Direct Target Sampling (参考) | 0.066 |
| **TreeSpark (Ours)** | **0.047** |
| Top-k + Naive Rejection (Control) | 0.462 |

> 表明 deterministic 扩展会严重扭曲分布，而 TreeSpark 严格保持 lossless。

---

### **消融实验结果**

| 实验 | 发现 |
|------|------|
| **Marginal Tree on DSpark** | T 仅为 2.60–3.33（$ N=28 $），远低于 chain（4.39），证明 marginal 构造在 semi-autoregressive 设置下失效 |
| **Uncalibrated Stopping Rule** | 使用 all-edges-fit 校准器导致 T 下降最多达 0.43，尤其在高 $ \theta $ 时更明显 |
| **Confidence Head as Priority** | 加入 confidence head 仅带来 +0.03–0.07 T 提升，且在采样下有害，最终被移除 |
| **Temperature-Scaled Input to Calibrator** | 导致树过大、性能下降，必须使用 unscaled $ q_o $ 输入校准器 |
| **With Replacement Sampling** | 导致采样树性能崩溃（T 从 4.94 降至 2.10），证实无放回必要性 |

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **Parent conditioning 是 semi-autoregressive draft tree 成功的关键**  
   - Marginal 构造在 co-trained drafter 上完全失败，说明必须利用 Markov head 提供的上下文依赖。

2. ✅ **校准后的路径生存概率是有效的价值函数**  
   - 两参数 Platt 校准器即可实现高精度边接受预测（ECE ≈ 0.01），且跨 domain、temperature、model scale 泛化良好。

3. ✅ **自适应停止优于任何固定预算配置**  
   - 在所有温度和负载下，adaptive $ \theta $ 控制的 TreeSpark 均位于“acceptance-vs-cost”前沿之上。

4. ✅ **采样过程必须保持 proposal 一致性**  
   - 无放回抽样 + 重建 proposal 是实现任意温度下 lossless 的关键。

5. ✅ **树大小应是运行时决策而非离线设定**  
   - 通过 $ \theta $ 暴露为调度接口，实现了负载感知的弹性伸缩，解决了“宽树在高并发下变慢”的根本矛盾。

---

### **局限性**
1. **Block Depth 限制**：当前 drafter 使用 $ b=7 $，限制了每轮最大接受长度；更深块需重新训练。
2. **Markov Conditioning 层次浅**：仅依赖 immediate parent，长距离依赖建模能力有限，随深度加深效果衰减。
3. **Greedy Labels for Calibration**：校准使用的是 greedy 匹配标签，而采样时使用 rejection rule，存在轻微错配（虽实验证明影响小）。
4. **尚未集成生产级 batching kernel**：当前动态批处理引擎为研究原型，未整合 production-grade CUDA kernels。

---

### **未来工作方向**
- 设计 **candidate-aware confidence head**，直接预测边接受率。
- 探索 **deeper block drafting** 与 **multi-step Markov modeling**。
- 将 TreeSpark 框架推广至其他具备 cheap conditional head 的 drafter（如 Domino, JetSpec）。
- 开发 **real-time feedback controller**，基于实时延迟信号动态调节 $ \theta $。
- 扩展至 **多模态 speculative decoding** 场景。

---

> 📦 **代码与资源**：https://github.com/PopSoda2002/TreeSpark  
> 所有 checkpoint、prompt、实验脚本均已开源，支持完整复现。

</details>

---

### 8. [Accelerating Dense LLMs via L0-regularized Mixture-of-Experts](https://arxiv.org/abs/2609.21672)

**Authors**: Zhenyu Zhang, Jiudong Yang, Zhaowen Tao, Meng Chen  
**Category**: cs.AI  
**Published**: 2026-09-22  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.21672v1  

#### Abstract
Large language models (LLMs) achieve strong performance but suffer from slow and costly inference. Existing acceleration methods often lead to noticeable performance degradation, while Mixture-of-Experts (MoE) models require extensive computational resources. In this paper, we propose L0-MoE, a ligh...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Accelerating Dense LLMs via L0-regularized Mixture-of-Experts

## 1. 论文的主要贡献和创新点

### 解决了什么问题
大型语言模型（LLMs）虽然在多项任务上表现出色，但其**推理速度慢、计算成本高**的问题严重制约了实际部署。现有的加速方法如量化（Quantization）、剪枝（Pruning）和知识蒸馏（Knowledge Distillation）通常会导致明显的性能下降。而 Mixture-of-Experts（MoE）模型虽能提升效率，但通常需要从头训练或大规模语料微调，计算资源消耗巨大。

本文旨在解决以下核心挑战：
- 如何在**小规模训练语料**（仅30B tokens）下，高效构建 MoE 模型以加速密集 LLM 推理；
- 如何在几乎不损失性能的前提下实现显著的推理加速。

---

### 提出了什么新方法或新思路
作者提出 **L0-MoE** —— 一种基于 **L0-regularization** 的轻量级 MoE 构建方法，包含三个关键技术组件：

1. **Cluster Confusion Matrix (CCM) 基于采样的数据集构建**
   - 利用 BGE-M3 编码器提取文本语义向量，结合 K-means 聚类划分语义域；
   - 引入“聚类混淆矩阵”评估不同迭代中聚类中心的变化，筛选出语义区分度更高的子数据集用于训练；
   - 实现**领域感知的数据集构建**，确保每个专家学习到特定语义领域的知识。

2. **基于 L0-regularization 的专家构造**
   - 在预训练 dense LLM 的 FFN 层中应用 L0 正则化，自动选择对特定领域最重要的隐藏维度；
   - 冻结非 MLP 参数，仅通过少量训练即可形成多个专业化“专家”；
   - 避免了传统 MoE 从零训练的巨大开销。

3. **动态批处理（Dynamic Batching）策略**
   - 设计两阶段调度机制：
     - 初期使用语义相近的样本进行 batch 构造，帮助路由机制快速初始化；
     - 后期引入跨领域样本组合，增强 token-level 的专家选择能力；
   - 显著提升了 MoE 训练效率与稳定性。

---

### 相比现有方法的优势
| 方法 | 是否需大规模训练 | 性能保留 | 推理加速比 | 所需资源 |
|------|------------------|----------|------------|-----------|
| GPTQ / AWQ（量化） | 否 | 中等下降 | ~1.8x | 低 |
| LLM-Shearing（剪枝） | 否 | 明显下降 | ~2.6x | 低 |
| RKD+CoT（蒸馏） | 是 | 显著下降 | ~5.1x | 中 |
| DeepSeek-MoE / Mixtral | 是 | 高 | 高 | 极高（万亿token） |
| **L0-MoE（本文）** | **否（仅30B tokens）** | **几乎无损** | **2.0–2.5x** | **低至中** |

> ✅ **优势总结**：
> - **极低训练成本**：仅需 30B tokens 和单次微调；
> - **高性能保持**：在多个基准上接近甚至略超原始模型；
> - **高效推理加速**：达到 2.5x 加速，优于多数轻量级优化方法；
> - **框架无关性**：可在 FSDP、SGlang、vLLM 等主流系统上运行。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **训练数据**：RedPajama 数据集的一个子集（共 30B tokens），通过 CCM 方法采样构建。
- **评估基准**（四大公开测试集）：
  - **MMLU**：多任务语言理解，涵盖 STEM、人文、社科等 57 个学科；
  - **GSM8K**：小学数学应用题，测试推理能力；
  - **HumanEval**：代码生成任务，评估编程能力；
  - **BigBench Hard (BBH)**：极具挑战性的复杂推理任务集合。

---

### 实验设置和评估指标
- **基础模型**：
  - Llama-3-8B
  - Mistral-7B
  - Qwen2-7B
- **MoE 设置**：
  - 专家数量 $ K = 64 $
  - 每个 token 激活 Top-2 专家
  - 底层若干层保持 dense（如 Qwen2 使用前 4 层）
- **训练配置**：
  - 使用 FSDP 进行分布式训练（Zero-3 分片）
  - 推理使用 SGlang 框架统一评测，保证公平性
  - 学习率：$1 \times 10^{-4}$，warmup 比例 0.2（专家）和 0.06（MoE 层）
- **评估指标**：
  - 准确率（Accuracy）在 MMLU、GSM8K、HumanEval、BBH 上的平均得分
  - 推理速度加速比（Speedup）

---

### 基线方法对比
- **原始模型**：Llama-3-8B、Mistral-7B、Qwen2-7B
- **加速方法**：
  - **GPTQ**：4-bit 量化
  - **LLM Shearing**：结构化剪枝
  - **RKD + CoT Distillation**：反向知识蒸馏 + 思维链蒸馏
- 所有基线均在同一推理框架（SGlang）下测试，确保可比性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & 2）

| 模型 | MMLU | GSM8K | HumanEval | BBH | 平均 | Speedup |
|------|------|-------|-----------|-----|--------|---------|
| Llama-3-8B | 66.6 | 56.0 | 33.5 | 57.7 | 53.5 | — |
| **L0-MoE (Llama-3)** | 66.3 | 55.9 | 33.7 | 57.2 | **53.3** | **2.0x** |
| Mistral-7B | 64.1 | 52.2 | 29.3 | 56.1 | 50.4 | — |
| **L0-MoE (Mistral)** | 64.8 | 53.6 | 31.1 | 55.9 | **51.4** | **2.1x** |
| Qwen2-7B | 70.3 | 79.9 | 51.2 | 62.6 | 66.0 | — |
| **L0-MoE (Qwen2)** | 70.4 | 80.5 | 52.0 | 61.5 | **66.1** | **2.5x** |

> 🔍 **观察**：L0-MoE 在所有模型上实现了 **2.0–2.5x 推理加速**，且性能基本持平，**Mistral 版本甚至略有提升（+1%）**

---

### 与基线方法对比（Qwen2-7B 为 backbone）

| 方法 | MMLU | GSM8K | Speedup |
|------|------|-------|---------|
| Qwen2-7B | 70.3 | 79.9 | — |
| **L0-MoE** | **70.4** | **80.5** | **2.5x** |
| GPTQ | 67.8 | 73.8 | 1.8x |
| LLM Shearing | 68.2 | 75.5 | 2.6x |
| RKD+CoT | 61.2 | 60.2 | 5.1x |

> 📌 **结论**：尽管 RKD+CoT 达到最高加速比（5.1x），但性能大幅下降；而 L0-MoE 在加速的同时**保持最强性能表现**。

---

### 消融实验结果（Ablation Study，Table 3）

| 模型变体 | MMLU | GSM8K |
|--------|------|-------|
| **Full L0-MoE** | **70.4** | **80.5** |
| CCM w/o K-means | 68.2 | 78.1 |
| w/ random order batching | 68.2 | 75.5 |
| w/ random batch batching | 66.6 | 77.1 |
| Random MoE | 48.1 | 69.6 |
| Magnitude Pruning | 52.6 | 69.1 |
| OBS | 68.4 | 74.1 |
| SVD | 55.2 | 73.8 |

> 🔍 **发现**：
> - 移除 K-means 或动态批处理导致明显性能下降 → 表明 **CCM 和动态批处理至关重要**；
> - 替换 L0-regularization 为其他方法（Random/Magnitude/OBS/SVD）均显著劣化 → 证明 **L0 正则化在专家构建中的优越性**。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **L0-regularization 可有效用于 MoE 构建**：无需从头训练，即可从 dense LLM 中提取出功能分离的专家模块。
2. **小规模语料也能构建高性能 MoE**：仅用 30B tokens，通过 CCM 和动态批处理，就能实现媲美原模型的性能。
3. **推理加速与性能保留可以兼得**：L0-MoE 实现了 **高达 2.5x 的推理加速**，同时在多个基准上**未见性能损失**，部分情况下还有轻微增益。
4. **动态批处理显著提升训练效果**：通过控制 batch 内语义分布，有效引导路由机制的学习过程。

---

### 方法的局限性
1. **暴露偏差（Exposure Bias）**：
   - 当前专家训练基于 sequence-level 聚类，但推理时是 token-level 路由，存在不一致性。
2. **专家冗余风险**：
   - 尚未显式衡量专家之间的差异性，可能导致功能重叠，影响加速效率。
3. **依赖语料多样性**：
   - CCM 效果依赖于预训练语料的 topic 多样性（如 RedPajama），在单一领域语料中可能失效。
4. **未与大规模 MoE 直接比较**：
   - 未对比 DeepSeek-MoE、Mixtral 等千亿 token 训练的 MoE 模型，适用场景不同。

---

### 未来工作方向
1. **探索 token-level 数据划分**：缓解 sequence-level 与 token-level 的不匹配问题。
2. **设计专家差异化学习机制**：引入正交约束或对比学习，减少参数冗余。
3. **扩展至更大模型（如 70B+）**：验证是否能在更大规模上获得更高加速比。
4. **扩大语料规模**：尝试在更多 token 上训练，进一步提升 L0-MoE 性能，超越 dense LLM。
5. **开源数据与代码**：作者承诺将发布 curated dataset 和 code，推动复现与改进。

---

> ✅ **总体评价**：  
> L0-MoE 是一项**实用性强、工程价值高的 LLM 加速方案**，特别适合资源受限但追求高性能推理的工业场景。它打破了“MoE 必须大规模训练”的固有认知，为轻量化 MoE 架构开辟了新路径。

</details>

---

### 9. [StepKV: Step-Aware KV Cache Compression for LLM Agents](https://arxiv.org/abs/2609.22158)

**Authors**: Boyu Feng, Jiahong Liu, Yifan Li, Wenhao Yu, Zexuan Qiu, Yuliang Sun, Ming Shen, Xiang Li, Quanyu Dai, Irwin King  
**Category**: cs.LG  
**Published**: 2026-09-22  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.22158v1  

#### Abstract
Key-value (KV) caching is essential for efficient autoregressive large language model (LLM) inference, but the cache grows linearly with context length, increasing storage and decoding costs. KV cache compression mitigates this cost by retaining only a subset of cached tokens. This challenge is part...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：StepKV: Step-Aware KV Cache Compression for LLM Agents**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
在 **LLM Agents** 的多步推理场景中，传统的 **KV Cache Compression** 方法通常将缓存视为扁平的 token 流，并基于 **recency** 或 **attention-based saliency** 进行 token 级别的剪枝。然而，这种做法忽略了 agent 推理过程中的 **结构化特性** —— 即信息是以“推理步骤”（reasoning step）为单位组织的。

这导致了一个关键问题：**Reasoning Continuity Disruption**（推理连续性中断）。即，在低 KV 缓存预算下，早期但关键的观察或中间决策可能因局部不活跃而被过早删除，从而破坏后续推理所需的上下文依赖。

### **提出了什么新方法或新思路**
论文提出 **StepKV**，一种**面向推理步骤的 KV Cache 压缩框架**，其核心思想是：

- 将 **reasoning step** 视为一级保留单元，而非仅关注单个 token。
- 引入 **step-level utility** 信号，结合 **token-level saliency**，共同决定 token 的保留优先级。
- StepKV 不是简单地保护整个步骤，而是通过 step utility **提升**该步骤内所有 token 的保留权重，同时仍允许细粒度选择。

具体机制包括三个阶段：
1. **New-step Scoring**：为每个完成的步骤计算初始得分 $ r_k $，综合考虑：
   - `Observation validity`（观察是否有效）
   - `Evidence gain`（是否引入新证据）
   - `Action redundancy`（动作是否重复）
2. **Previous-step Updating**：维护一个 `reuse accumulator` $ c_k $，当后续步骤复用某步骤的观察时，动态提升其重要性。
3. **Step Utility & Cache Selection**：最终 step utility $ S_k = \text{clip}(w_r r_k + w_c \log(1 + c_k)) $，并与 token saliency $ T_i $ 结合得到综合评分：
   $$
   P_i = \alpha T_i + \beta S_k
   $$
   按此全局排序并保留 top-B tokens。

### **相比现有方法的优势**
- **更符合 agent 推理结构**：显式建模 step-level 重要性，避免关键推理路径被碎片化。
- **更强的鲁棒性**：在极低 KV 预算（如 20%）下仍能保持较高准确率，而 token-level 方法严重退化。
- **效率与精度更好权衡**：在显著减少 KV Cache 大小和推理延迟的同时，维持甚至超过 Full KV 的任务表现（尤其在长程任务中）。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **HotpotQA**：多跳问答，衡量 **Exact Match (EM)** 和 **F1**。
- **2WikiMultihopQA**：类似 HotpotQA，用于验证泛化性。
- **MuSiQue**：构造的多跳问题，强调推理链完整性。
- **BrowseComp-Plus**：**长程网页推理基准**，轨迹可超 100K tokens，评估极端长上下文下的效率与效果。

### **实验设置和评估指标**
- **Backbone Models**：
  - Qwen2.5-7B-Instruct
  - Llama-3.1-8B-Instruct
- **KV Keep Ratios**：50%、20%
- **评估指标**：
  - **任务性能**：EM、F1
  - **效率指标**：KV Cache Size、Inference Latency、Peak/Avg Cache Usage
- **随机种子**：3 次运行取均值 ± 标准差

### **基线方法对比**
| 方法 | 类型 | 代表机制 |
|------|------|--------|
| **Full KV** | 上限参考 | 不压缩，完整缓存 |
| **ReAct** | 原始框架 | 无 KV 控制 |
| **H2O** | Accumulated Score-based | 基于累计 attention 保留重 token |
| **TOVA** | Online Attention-based | 动态保留高 attention token |
| **TokenSkipping** | Heuristic-based | 固定规则跳过 token |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### **表1：多跳 QA 数据集上的主结果（部分摘录）**

| 方法 | 预算 | HotpotQA EM (Qwen) | 2Wiki EM (Qwen) | MuSiQue EM (Qwen) |
|------|------|---------------------|------------------|--------------------|
| Full KV | 100% | 26.40 | 24.20 | 5.13 |
| H2O | 50% | 12.00 | 10.67 | 1.93 |
| TOVA | 50% | 10.07 | 9.20 | 1.93 |
| **StepKV (Ours)** | **50%** | **23.23** | **22.67** | **7.27** |
| H2O | 20% | 1.53 | 1.07 | 0.00 |
| TOVA | 20% | 0.73 | 1.47 | 0.07 |
| **StepKV (Ours)** | **20%** | **18.07** | **13.47** | **2.80** |

> ✅ **结论**：StepKV 在所有设置下均为最佳压缩方法，尤其在 20% 预算下优势巨大。

#### **BrowseComp-Plus 上的效率表现（图4）**
- StepKV 显著降低 **最大 KV Cache** 和 **推理延迟**（平均下降 >50%）。
- 在某些情况下，**EM/F1 甚至超过 Full KV**，说明原始缓存中存在冗余干扰。

---

### **与基线方法的对比结果**
- 在 **20% KV 预算** 下：
  - H2O/TOVA 的 EM 接近 **0**，F1 极低。
  - StepKV 保持 **18–20 EM**（HotpotQA），约为 Full KV 的 70%。
- 在 **50% 预算** 下：
  - StepKV 接近 Full KV 表现，而 token-level 方法损失超过 50% 性能。
- **推理开销**：
  - Token-level 方法因推理中断导致重复探索，反而 **KV Cache 膨胀**。
  - StepKV 维持推理连贯性，**实际缓存更小、延迟更低**。

---

### **消融实验结果**

#### **表2：Step Score 各组件消融（MuSiQue）**
| 变体 | EM↓ | F1↓ |
|------|-----|-----|
| Full StepKV | 7.00 | 12.49 |
| w/o reuse update | -3.00 | -6.45 |
| w/o evidence gain | -3.40 | -5.47 |
| w/o validity | -2.80 | -5.58 |
| w/o redundancy | -2.40 | -5.52 |

> 🔍 所有组件均有贡献，**reuse update** 和 **evidence gain** 最关键。

#### **表3：Step Score vs Token Score 消融**
| 变体 | EM↓ | F1↓ |
|------|-----|-----|
| w/o step score | -5.60 | -7.30 |
| w/o token score | +0.20 / -2.33* |

> 📌 **移除 step score 导致大幅下降**，说明 step-level 信号至关重要；  
> 移除 token score 对 EM 影响小但 F1 下降，表明 token-level 选择仍有必要。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Reasoning Continuity 是 KV 压缩的关键挑战**：传统 token-level 压缩破坏推理路径，导致性能崩溃。
2. **Step-level 信号显著提升压缩质量**：通过建模 step utility，StepKV 能识别并保护对最终答案有长期影响的推理步骤。
3. **StepKV 实现更优的效率-精度权衡**：在极低缓存预算下仍保持可用性能，适用于长程 agent 任务。
4. **长推理深度加剧压缩难度**：越深的推理链越依赖早期信息，StepKV 在此类场景中优势更明显（见图6）。

---

### **方法的局限性**
- **依赖外部轨迹信号**：step utility 基于 observation、action 等文本字段计算，若工具返回噪声或信息隐含表达，估计可能不准。
- **未利用模型内部表示**：当前方法无需训练，但可能不如基于 hidden states 或 representation learning 的方法精准。
- **评估范围有限**：目前仅在中等规模模型（7B/8B）和特定 agent 任务上验证，尚未扩展到更大模型或多模态 agent。

---

### **未来工作方向**
- **更精确的 step utility 估计**：结合模型内部状态（如 thought vectors）、注意力模式或强化学习信号。
- **动态调整 step/token 权重**：根据任务复杂度自适应调节 $ \alpha/\beta $。
- **扩展至多模态与复杂工具调用场景**：支持图像、代码执行等非文本 step。
- **与外部 memory 系统协同优化**：将 KV Cache 与 Memory Bank、MemGPT 等机制联合设计。

---

> **总结**：StepKV 提出了一种**以推理步骤为中心的 KV Cache 压缩范式**，通过融合 step-level utility 与 token-level saliency，在多步 LLM Agent 场景中实现了**更强的鲁棒性和更高的效率-精度比**，为长上下文 agent 推理提供了实用且有效的解决方案。

</details>

---

### 10. [WaveFront Decoding: Parallelized Self-Speculative Decoding for Looped Language Models](https://arxiv.org/abs/2609.23033)

**Authors**: Hyeongju Ha, Jae-Joon Kim  
**Category**: cs.LG  
**Published**: 2026-09-22  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.23033v1  

#### Abstract
Looped language models repeatedly apply a weight-shared block to increase effective depth without increasing parameter count, but the resulting T sequential recurrent-block calls per generated token substantially increase decoding latency. To address the issue, we introduce Wavefront Decoding (WFD),...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Wavefront Decoding: Parallelized Self-Speculative Decoding for Looped Language Models*

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

**Looped Language Models**（如 Ouro 和 Huginn）通过重复应用一个共享的 `recurrent block` 来增加有效深度，从而在不增加参数量的前提下提升模型能力。然而，这种架构在 **decoding 阶段存在高延迟问题**：生成每个 token 需要进行 $T$ 次顺序的 `recurrent block` 调用，导致解码过程严重受限于内存带宽。

此外，虽然已有工作提出利用浅层输出作为 draft（草案），深层输出作为 verifier（验证器）的 **self-speculative decoding** 思路（即 Draft-then-Verify, DtV），但其仍采用 **分阶段执行**（先 draft 后 verify），未能充分利用并行性。

---

### **提出了什么新方法或新思路**

本文提出 **Wavefront Decoding (WFD)** ——一种无需训练、适用于所有 looped LM 架构的 self-speculative decoding 框架。

#### 核心思想：
- 利用 looped LM 的两个特性：
  1. 中间 recurrence 输出可作为有效的 draft 预测；
  2. 参数共享允许不同位置、不同 recurrence 深度的状态在一次 batched 推理中统一处理。
- 将多个 token 的状态组织成一个 **对角波阵面（diagonal wavefront）**，在同一轮 `recurrent block` 调用中同时推进：
  - 新 token 在浅层 depth 进行 drafting；
  - 旧 token 在深层 depth 继续推进至 full-depth 并进行 verification。
- 实现 drafting 与 verification 的 **融合流水线**，而非分离阶段。

#### 算法特点：
- **无额外训练需求**，不依赖外部 draft model；
- 支持 full-stack 和 P/R/C 两类 looped 架构；
- 错误 speculation 由 full-depth verifier 自动纠正，保证输出与 autoregressive decoding 一致。

---

### **相比现有方法的优势**

| 特性 | Autoregressive (AR) | Draft-then-Verify (DtV) | Wavefront Decoding (WFD) |
|------|---------------------|--------------------------|----------------------------|
| 是否顺序执行 | 是 | 分阶段顺序 | ✅ 流水线并行 |
| 是否复用 draft 计算 | 否 | ✅ 是 | ✅ 是 |
| 是否共批处理 drafting & verification | ❌ 否 | ❌ 否 | ✅ 是 |
| 对 rejection 的容忍度 | — | 差（浪费整个 draft block） | ✅ 好（仅刷新 wavefront 内部分状态） |
| 加速潜力 | 基线 | 受限于 draft block 长度权衡 | ✅ 更高且更稳定 |

> ✅ WFD 首次实现了在预训练 looped LMs 中跨 token 位置与 recurrence depth 的连续 co-batching。

---

## 2. 核心实验方法和设置

### **使用的模型与任务类别**

- **主模型**：
  - **Ouro-2.6B**（full-stack 类型，$T=4$）
  - **Huginn-3.5B**（P/R/C 类型，$T=32$）
- **任务类别**（基于 Spec-Bench 协议）：
  - Multi-turn Conversation
  - Translation
  - Summarization
  - Question Answering
  - Mathematical Reasoning
  - Retrieval-augmented Generation

---

### **实验设置**

- **硬件平台**：单张 NVIDIA RTX A6000，使用 bfloat16 精度，greedy decoding。
- **输入配置**：
  - 用户 batch size 默认为 1；
  - Prompt 长度上限 1,024 tokens；
  - 生成长度上限 512 tokens。
- **评估指标**：
  - **Throughput (TPS)**：总生成 token 数 / 总 wall-clock 时间（含 prefill 和 decoding）；
  - **Speedup**：相对于 autoregressive decoding 的加速比；
  - **Acceptance Rate ($\alpha$)**：speculative draft 被接受的比例；
  - **Task Accuracy**：在 GSM8K 和 MATH-500 上的准确率。

---

### **基线方法对比**

| 方法 | 描述 |
|------|------|
| **Autoregressive (AR)** | 标准逐 token 生成，每步调用 $T$ 次 recurrent block |
| **Draft-then-Verify (DtV)** | 先生成 $y$ 个浅层 draft（depth $T_d < T$），再批量验证 |
| **WFD (ours)** | 提出的方法，混合 depth 批处理，持续 drafting + verification |
| **WFD + Cross-recurrence KV Sharing** | 结合 KV 缓存共享以降低长上下文开销 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### 表格汇总（来自 Table 1）：

| Model | Method | Overall Speedup | Acceptance Rate ($\alpha$) |
|-------|--------|------------------|-----------------------------|
| Ouro-2.6B | AR | 1.00× | — |
| Ouro-2.6B | DtV | **1.90×** | 0.92 |
| Ouro-2.6B | WFD | **2.42×** | 0.92 |
| Huginn-3.5B | AR | 1.00× | — |
| Huginn-3.5B | DtV | **2.79×** | 0.94 |
| Huginn-3.5B | WFD | **3.54×** | 0.94 |

> ✅ WFD 在两个模型上均显著优于 DtV，分别实现 **2.42× 和 3.54×** 的端到端吞吐提升。

---

### **与基线方法的对比结果**

- **全面超越 DtV**：
  - 在所有六类任务中，WFD 均优于 DtV；
  - 相对 DtV 进一步提速 **1.27×**；
  - 在低 acceptance 场景下优势更明显（例如 Ouro 翻译任务中 DtV 仅 1.06×，而 WFD 达 1.92×）。

- **原因分析**：
  - DtV 的 rejection 会丢弃整个未被接受的 draft block，造成大量计算浪费；
  - WFD 的 rejection 仅清除 wavefront 中最多 $W-1$ 个 speculative token，恢复成本更低。

---

### **消融实验结果**

#### （1）**Cross-recurrence KV Sharing 的影响**（Table 2）

| Model | Method | KV Sharing | Speedup (GSM8K) | Accuracy |
|-------|--------|------------|------------------|----------|
| Huginn-3.5B | AR | No (32 slots) | 1.00× | 30.64 |
| Huginn-3.5B | WFD | No | 2.54× | 31.12 |
| Huginn-3.5B | AR | Yes (4 slots) | 1.00× | 31.12 |
| Huginn-3.5B | WFD | Yes | **4.81×** | 30.59 |
| Huginn-3.5B | WFD | 1 slot | **4.77×** | 31.16 |

> 🔺 结合 KV sharing 后，WFD 在 Huginn 上达到 **最高 4.81× 加速**，且精度基本不变。

#### （2）**可控调度分析**（Figure 3）

- 控制 acceptance rate $\alpha$ 从 0.6 到 0.95：
  - WFD 在整个范围内始终优于所有 DtV 配置（$y=2$ 到 $y=32$）；
  - DtV 存在明显的 trade-off：长 draft block 提升高 $\alpha$ 下性能，但在低 $\alpha$ 下性能骤降；
  - WFD 不需调节 $y$，性能鲁棒性强。

#### （3）**上下文长度扩展性**（Figure 6）

- 随着 prefill 长度增长：
  - 无 KV sharing 时，WFD 的 speedup 明显下降（因 wavefront 内各 depth 访问不同 KV cache，traffic 随宽度 $W$ 增加）；
  - 引入 **cross-recurrence KV sharing** 后，WFD 在长达 64k tokens 的上下文中仍能维持约 **4.4× 加速**。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **Wavefront Decoding 成功打破了 drafting 与 verification 的相位隔离**，首次实现在 looped LMs 中对 mixed-depth token states 的连续 co-batching。
2. ✅ WFD 在真实场景中实现 **2.42×～3.54× 的端到端加速**，显著优于 DtV，尤其在低 acceptance 或复杂任务中表现更优。
3. ✅ **Cross-recurrence KV sharing 是释放 WFD 全部潜力的关键**，可将加速比进一步推高至 **4.81×**，同时保持 accuracy 稳定。
4. ✅ WFD 对 acceptance rate、batch size 和 context length 具有更强鲁棒性，无需手动调参 draft block 长度。

---

### **方法的局限性**

1. 🚫 **长上下文下的 KV traffic 瓶颈**：
   - 若不启用 KV sharing，wavefront 宽度 $W$ 导致 KV 缓存访问量线性上升，限制加速效果。
2. 🚫 **依赖特定架构特性**：
   - 仅适用于具有 weight-sharing recurrence 结构的 looped LMs，无法直接用于 standard Transformer。
3. 🚫 **当前评估限于 greedy decoding 和单卡环境**：
   - 未测试采样策略（sampling）、多 GPU 分布式部署等实际场景。

---

### **未来工作方向**

1. ✅ 探索 **tree-structured drafting** 在 WFD 中的应用，进一步提高 MAT（Mean Accepted Tokens）；
2. ✅ 结合 **adaptive recurrence depth** 或 **adaptive draft depth** 策略，动态调整 $T_d$ 以优化 trade-off；
3. ✅ 扩展至 **distributed P/R/C execution** 和 **multi-GPU setting**，支持更大规模部署；
4. ✅ 研究如何在 **非 looped 模型** 上模拟 wavefront 调度机制（如通过 layer-skipping 或 early exiting）。

---

> 💡 **总结一句话**：  
> **Wavefront Decoding 通过“对角波阵面”调度，将 self-speculative decoding 从分阶段推向连续流水线，在无需训练的情况下为 looped LMs 带来高达 4.81× 的推理加速，是迈向高效 latent reasoning 推理的重要一步。**

</details>

---

### 11. [H-Spec: Parallel Speculative Decoding Without a Drafter-Side KV Cache](https://arxiv.org/abs/2609.24197)

**Authors**: Weifan Jiang, Krishna Teja Chitty-Venkata, Megan Flynn, Reed Meyerson, Zhenting Qi, Tianyu Wu, Eldar Kurtic, Minlan Yu, Alexandre Marques  
**Category**: cs.LG  
**Published**: 2026-09-22  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.24197v1  

#### Abstract
Speculative decoding losslessly accelerates large language model inference by having a lightweight draft model predict future tokens for verification by the target model. Recent block diffusion drafters further reduce drafting latency by predicting multiple tokens in parallel. However, existing bloc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# H-Spec: Parallel Speculative Decoding Without a Drafter-Side KV Cache 论文总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有的 **block diffusion drafters**（如 DFlash、DSpark）通过并行预测多个 token 来加速大语言模型（LLM）推理，但它们依赖于将目标模型的隐藏状态投影到一个独立的 **drafter-side KV cache** 中以注入上下文。这一机制带来了以下问题：

- **内存开销高**：每个请求都需要维护额外的 KV cache，导致 GPU 内存占用显著增加（1.1–1.8×）。
- **写入延迟随并发增长**：在高并发服务场景下，KV 注入的时间开销呈线性上升，限制了吞吐量扩展。

因此，本文提出核心问题：  
> 能否设计一种无需额外 **drafter-side KV cache** 的并行 drafter，同时保持高质量的 draft 预测？

---

### 🚀 提出的新方法与创新思路

作者提出了 **H-Spec** —— 一种新型混合架构的并行 drafter，其核心是 **Hybrid Target-Context Injection（混合目标上下文注入）** 机制：

#### （1）混合上下文注入（Hybrid Context Injection）
不再单独构建 drafter-side KV cache，而是结合两种互补的目标模型上下文源：
- **in-place target KV reuse**：直接复用目标模型已有的 KV cache，提供细粒度的位置级信息。
- **last-token target hidden states**：仅使用最后一个输入 token 的隐藏状态，经融合后作为 Mamba 模块的初始状态，提供完整的前缀摘要。

该方法避免了 $O(N)$ 的额外 KV 存储，仅引入 $O(1)$ 开销。

#### （2）H-Spec 架构设计
基于上述思想，构建了一个 **hybrid Mamba-attention** 并行 drafter：
- **Mamba 模块**：接收 last-token hidden states 初始化其 recurrent state，利用其对全局上下文建模的能力。
- **Attention 模块**：直接重用目标模型的 KV cache，捕捉位置相关依赖。
- **并行扫描支持 block-parallel drafting**：尽管 Mamba 是递归结构，但通过 parallel scan 实现所有 draft token 的并行生成。

此外，保留轻量级 **causal correction head**（如 DSpark 的 Markov head），增强相邻 token 间的连贯性。

---

### 🔍 相比现有方法的优势

| 维度 | H-Spec | 传统 Block Drafters（如 DFlash） |
|------|--------|-------------------------------|
| **KV Cache 开销** | ❌ 无额外 drafter-side KV cache | ✅ 需要独立 KV cache（+13.9%-80% 内存） |
| **并发可扩展性** | ✔️ KV 写入开销不随并发增长 | ❌ KV 注入时间随并发线性上升 |
| **draft 质量** | ⬆️ 更高的 mean accepted length（MAL） | ⬇️ 后期位置 accept rate 下降明显 |
| **系统效率** | ⬆️ 更高 throughput，更低 KV 利用率 | ⬇️ 高并发时受限于 KV 写瓶颈 |

---

## 2. 核心实验方法和设置

### 📚 数据集与任务
- **训练数据**：从 Magpie 和 Ultrachat 中采样 100K 样本（比例 60:40），用于训练所有 drafter。
- **评估任务覆盖 8 大领域**：
  - 数学（GSM8K、MATH）
  - 编程（HumanEval）
  - 对话（MT-Bench、Alpaca）
  - RAG、摘要、翻译、工具调用（BFCL）
- 扩展评估还使用了 DSpark 发布的 **DeepSpec 9-task suite**。

---

### ⚙️ 实验设置
- **目标模型**：
  - `Llama3.1-8B-Instruct`
  - `Qwen3-4B`, `Qwen3-8B`
- **每步 draft token 数**：$k=7$
- **批大小**：batch size 1（单请求延迟）、并发请求 $C \in \{8,16,...,128\}$（吞吐测试）
- **硬件平台**：NVIDIA A100 80GB GPU
- **服务框架**：vLLM + PagedAttention
- **统一训练流程**：使用 Speculators 框架复现所有 baseline，确保公平比较。

---

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Mean Accepted Length (MAL, $\bar{t}$)** | 每个 speculative step 平均接受的 draft token 数量，越高越好 |
| **Inter-Token Latency (ITL) Speedup** | 相比无 speculative decoding 的端到端延迟加速比 |
| **Throughput (output tokens/sec)** | 高并发下的系统吞吐能力 |
| **KV Cache Utilization (%)** | GPU 上 KV cache 占用比例，越低越好 |

---

### 🔁 基线方法对比
- **P-Eagle**：基于共享隐状态的并行 drafter
- **DFlash**：典型 block diffusion drafter，使用独立 KV cache
- **DSpark**：DFlash 改进版，加入 confidence scheduling 和 Markov head
- **DFlash-2**（补充对比）：最新改进版本，引入卷积和路径选择器

---

## 3. 主要实验结果和性能指标

### 📈 单请求性能（Batch Size = 1）

| 目标模型 | 最佳 baseline MAL | **H-Spec MAL** | **提升幅度** | 最佳 baseline Speedup | **H-Spec Speedup** | **提升幅度** |
|---------|------------------|---------------|-------------|------------------------|--------------------|-------------|
| Llama3.1-8B | 2.72 | **3.06** | **+12.5%** | 2.23x | **2.50x** | **+12.1%** |
| Qwen3-4B | 3.09 | **3.24** | **+4.8%** | 2.37x | **2.50x** | **+5.5%** |
| Qwen3-8B | 2.95 | **3.21** | **+8.8%** | 2.40x | **2.61x** | **+8.8%** |

> ✅ 在三个目标模型上，H-Spec 平均将 MAL 提升 **5.0–13.3%**，ITL 加速比提升 **5.3–12.6%**

---

### 📦 高并发服务性能（vLLM）

| 场景 | 指标 | 结果 |
|------|------|------|
| **最大吞吐量** | vs. DSpark | 提升 **5.1–17.3%** |
| **KV Cache 利用率** | vs. 最优 baseline | 降低 **4.3–24.6%** |
| **并发扩展性** | C=8 → C=128 | 吞吐优势持续扩大（1.13–1.58×） |

> 💡 图 1 显示，在 8–128 并发请求下，H-Spec 在 **throughput-KV utilization 曲线上完全帕累托占优（Pareto-dominates）** 其他方法。

---

### 🔍 消融实验结果

#### （1）上下文来源消融（Ablation on Context Sources）
| 变体 | MAL 下降幅度（vs. 完整 H-Spec） |
|------|-------------------------------|
| Only last-token hidden states | ↓ 24.1–30.5% |
| Only in-place KV reuse | ↓ 7.1–11.7% |
| **完整 hybrid 注入** | ✅ 最优 |

> ✅ 证明两种上下文源**互补而非替代**，缺一不可。

#### （2）架构消融（Ablation on Drafter Architecture）
| 架构 | MAL 提升（vs. 单一模块） |
|------|--------------------------|
| Mamba-only | — |
| Attention-only | — |
| **Hybrid (Mamba + Attention)** | ↑ 28.9–45.1% vs. Mamba-only<br>↑ 13.3–15.3% vs. Attention-only |

> ✅ 混合骨干优于任一单一结构。

#### （3）运行时扩展性分析
- **低并发 / 短上下文**：H-Spec 稍慢（更深计算图）
- **高并发 / 长上下文（>512 tokens）**：H-Spec 更快，因规避了 KV 注入瓶颈

> ✅ H-Spec 更适合生产环境中的长文本、高并发场景。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **drafter-side KV cache 是可避免的系统瓶颈**：它带来显著内存与延迟开销，尤其在高并发下成为性能瓶颈。
2. **混合上下文注入有效且高效**：结合 last-token hidden states 与 in-place KV reuse，可在零额外 KV 存储下维持甚至提升 draft 质量。
3. **Mamba + Attention 混合架构可行且优越**：Mamba 天然适配全局状态初始化，Attention 有效利用位置信息，二者协同提升 MAL。
4. **H-Spec 在真实部署中表现更优**：不仅单请求性能领先，在 vLLM 高并发服务中实现最高 throughput 与最低 KV 占用。

---

### ⚠️ 局限性
1. **KV dimension 匹配约束**：要求 drafter 的 K/V 维度必须与目标模型一致，限制了灵活性。
2. **低并发下略有性能损失**：由于混合架构更深，在极低负载时略逊于简单 block diffusion 模型。
3. **未探索部分 KV head 复用**：目前复用全部 KV heads，未来可通过子集复用进一步优化 attention 开销。

---

### 🔮 未来工作方向
1. 探索更灵活的 **target-layer selection 策略**（非均匀间隔层）
2. 研究 **partial KV head reuse** 以减少 attention 开销
3. 尝试其他 **causal correction 机制**（如 DFlash-2 的 path selector）替换 Markov head
4. 将 H-Spec 思路推广至 **multi-token prediction without a drafter** 范式（如 Medusa）

---

## 总结

> **H-Spec 成功解耦了高质量 draft 生成与高昂 KV 存储成本之间的强绑定关系**。它通过创新的 hybrid context injection 和 Mamba-attention 混合架构，在彻底消除 drafter-side KV cache 的前提下，实现了更高的 draft 接受率、更快的推理速度以及更强的并发服务能力。这项工作为未来高效、可扩展的 speculative decoding 系统设计提供了重要范式。

</details>

---

### 12. [Block-Sparse Attention with Semantic-Geometric Decoupled Routing](https://arxiv.org/abs/2609.22884)

**Authors**: Xinwei Long, Weigao Sun, Weibo Gao, Pengkun Jiao, Biqing Qi, Feida Zhu, Yiran Zhong, Steven Hoi, Bowen Zhou  
**Category**: cs.CL  
**Published**: 2026-09-22  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.22884v1  

#### Abstract
Long-context inference has become a defining capability of large language models, but exact dense attention remains costly due to its quadratic scaling with sequence length. Block-sparse attention offers a hardware-friendly alternative by routing each query block to a small set of relevant key block...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Block-Sparse Attention with Semantic-Geometric Decoupled Routing

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **long-context inference** 中，标准的 dense self-attention 因其 $O(L^2)$ 的计算复杂度成为性能瓶颈。虽然 **block-sparse attention** 通过将序列划分为块并仅关注相关 key blocks 来提升效率，但现有的训练无关（training-free）block routing 方法存在以下问题：

- 多数方法在应用 **RoPE**（Rotary Position Embedding）后对 token 表示进行池化（post-RoPE pooling），导致：
  - **语义聚合** 与 **位置几何** 耦合；
  - 高频维度因相位抵消而衰减（destructive interference），丢失局部位置信息；
  - 影响 block routing 的准确性。

为补偿损失，现有方法依赖 **token-level search** 或 **post-hoc 温度校准**，增加了推理开销或噪声放大风险。

---

### 🆕 提出的新方法：Semantic-Geometric Decoupled Routing (SGDR)

SGDR 是一种 **无需训练的 block routing 框架**，核心思想是 **解耦语义与几何建模**：

1. **语义聚合移至 pre-RoPE 空间**  
   在 RoPE 编码前对 block 内 token 进行均值池化，避免旋转后的高频向量平均造成的相位干扰，保留原始语义强度。

2. **几何偏置通过离线先验重建**  
   利用一个 **offline structural prior** 和 **相对块距离**（relative block distances）重构位置偏差，显式建模几何关系。

3. **得到闭式（closed-form）block routing score**  
   结合在线语义激活与离线几何先验，推导出可直接用于推理的 block-level 得分公式，无需 token 级搜索或后期调参。

---

### 🔍 相比现有方法的优势

| 优势 | 说明 |
|------|------|
| **高效性** | 路由开销极低（<3.4ms），支持硬件友好实现（定制 Triton kernel） |
| **高精度** | 在 4K–128K 上接近 full-attention 性能，优于多数 sparse baseline |
| **免训练 & 免搜索** | 完全 training-free，不依赖 token-level relevance search |
| **理论严谨** | 基于宽平稳过程假设和频域分析，提供数学推导支持 |

---

## 2. 核心实验方法和设置

### 📚 数据集

论文在四类长上下文任务上进行全面评估：

| 任务类型 | 数据集 | 描述 |
|--------|-------|------|
| **长文本检索** | RULER (Hsieh et al., 2024) | 测试从长输入中提取事实的能力 |
| **长文本理解** | LongBench (Bai et al., 2024) | 多任务、多语言长文档推理基准 |
| **长语言建模** | PG19 (Rae et al., 2019) | 衡量长文本预测能力（以 PPL 为指标） |
| **视频推理** | VideoMME (Fu et al., 2025), LongVideoBench (LVB) (Wu et al., 2024) | 视频问答与多模态理解 |

---

### ⚙️ 实验设置与评估指标

| 设置项 | 说明 |
|-------|------|
| **模型架构** | Llama-3.1-8B-Instruct, Qwen3-8B, Qwen3-VL-8B-Instruct |
| **上下文扩展** | 使用 YaRN 将 Qwen3-8B 上下文从 32K 扩展至 128K |
| **RoPE 变体** | 支持标准 RoPE、YaRN、Interleaved M-RoPE |
| **块大小** | $B=128$（遵循 Prism 设置） |
| **top-p 阈值** | $p=0.9$ |
| **评估指标** | 准确率（accuracy）、平均得分、Perplexity (PPL)、延迟（latency）、加速比（speedup） |

---

### 🆚 基线方法对比

| 基线方法 | 类型 | 是否需训练 | 特点 |
|---------|------|------------|------|
| **FlashAttention-2** | Dense Attention | 否 | 全注意力基线 |
| **StreamingLLM** | Sparse | 否 | 固定 attention sink |
| **MInference** | Sparse | 否 | 动态稀疏预填充，依赖 token-level search |
| **FlexPrefill** | Sparse | 否 | 上下文感知稀疏机制 |
| **XAttention** | Block-sparse | 否 | 抗对角线评分机制 |
| **Prism** | Block-sparse | 否 | 频谱感知 block routing，使用温度校准 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### ✅ **RULER 检索任务（表1）**

| 方法 | 平均得分（Llama-3.1） | 平均得分（Qwen-3-8B） |
|------|------------------------|------------------------|
| FlashAttn | 88.94 | 86.61 |
| Prism | 87.54 | 85.27 |
| **SGDR (Ours)** | **87.55** | **86.09** |

- 在 128K 上显著优于最强基线：
  - Llama-3.1: **+0.82** over Prism
  - Qwen-3-8B: **+0.69** over FlexPrefill

#### ✅ **LongBench 理解任务（表2）**

| 方法 | 平均得分（Llama-3.1） | 平均得分（Qwen-3-8B） |
|------|------------------------|------------------------|
| FlashAttn | 41.47 | 39.49 |
| Prism | 41.08 | 39.12 |
| **SGDR (Ours)** | 40.73 | **39.12** |

- 在 Qwen-3-8B 上与 Prism 持平，且无需 token-level search；
- 在 Few-shot Learning 子任务上甚至 **超过 full model**（58.60 vs 56.69）。

#### ✅ **语言建模（PG19，表3）**

| 方法 | 平均 PPL |
|------|----------|
| FlashAttn | 25.25 |
| Prism | 25.25 |
| **SGDR (Ours)** | **25.23** |

- PPL 略优，表明生成质量几乎无损；
- 随上下文增长稳定下降，符合 full attention 缩放规律。

#### ✅ **视频推理（表4）**

| 方法 | VideoMME (Overall) | LVB (Overall) |
|------|--------------------|---------------|
| FlashAttn | 71.22 | 65.00 |
| Prism | 71.22 | 64.25 |
| **SGDR (Ours)** | 70.85 | **64.96** |

- 在 LVB 上仅落后 full model **0.04 pts**，表现稳健。

---

### ⚡ 效率结果（图2 & 图3）

| 指标 | 结果 |
|------|------|
| **路由开销（Estimation Time）** | < **3.4 ms**（所有长度下几乎恒定） |
| **128K 上总 attention 延迟** | **89.59 ms**（FlashAttn: 451.03 ms） |
| **加速比（Speedup @128K）** | **5.03× over FlashAttn** |
| 对比其他方法 | 显著优于 MInference (4.79×) 和 Prism (3.76×) |

> 图3显示：XAttention 和 FlexPrefill 的估计时间随长度剧增（>80ms），而 SGDR 几乎无增长。

---

### 🔬 消融实验（表5，RULER 上 Qwen-3-8B）

| 变体 | 平均准确率 |
|------|-----------|
| **完整模型（Ours）** | **86.09** |
| w/o Offline Profiling ($A_f,\theta_f=1,0$) | 78.87 |
| w/o Macro Geometry ($\omega_f B \Delta M=0$) | 73.84 |
| w/o Online Phase ($\phi_f=0$) | 74.94 |
| w/o 所有位置信息（Semantic Only） | 67.26 |

- 所有组件均不可或缺，尤其 **offline profiling** 和 **macro geometry** 对长上下文至关重要；
- 移除位置信息后性能严重下降，验证了几何建模的重要性。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Post-RoPE pooling 引发高频特征衰减** 是 block routing 不准的根本原因；
2. **将语义池化前移至 pre-RoPE 空间** 可有效避免相位干扰，保护局部结构；
3. **结合 offline structural prior 与宏观距离** 可精确恢复几何偏置，形成闭式路由得分；
4. SGDR 在 **精度、效率、通用性** 上取得平衡，是首个实现“免搜索 + 高精度 + 极低开销”的 block-sparse 路由方案。

---

### ⚠️ 局限性

1. **仅适用于 Softmax-based Attention**  
   不兼容非 attention 架构（如 Gated Delta Net）或线性 attention 模型（如 Lightning Attention）。

2. **依赖 RoPE 几何结构**  
   虽然支持 RoPE 及其变体（如 YaRN），但无法直接应用于传统绝对位置编码（absolute PE）模型。

---

### 🔮 未来工作方向

- 探索在 **world models** 和 **video generation** 中的应用；
- 扩展至更多 position encoding 方案（如设计适配器兼容绝对 PE）；
- 结合 KV Cache 压缩技术进一步优化端到端推理延迟。

---

> **总结一句话**：  
> SGDR 通过 **语义-几何解耦**，实现了无需训练、无需 token 搜索、超低延迟且高精度的 block-sparse attention routing，在 128K 上达到 **5.03× FlashAttention 加速**，同时保持接近 full attention 的性能。

</details>

---

### 13. [FLARE: A Full-Lifecycle Dense Supervision Paradigm for Long-Horizon Coding Agents via Generative Reward Model](https://arxiv.org/abs/2609.23808)

**Authors**: Jingxuan Xu, Gang Wu, Yanan Wu, Yutao Mou, Songwei Yu, Tianzhuang He, Zhengshuo Gong, Zhao Liu, Zihang Xu, Wenqiang Zhu, Xinping Lei, Weihao Li, Yuhui Bai, Zhongqiu Wang, Yan Wu, Ariel Deng  
**Category**: cs.CL  
**Published**: 2026-09-22  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.23808v1  

#### Abstract
While test-time scaling enhances Large Language Model (LLM) agents in long-horizon software engineering (SWE), sparse binary rewards (Pass/Fail) create a severe credit assignment crisis and waste failed exploratory trajectories. Current trajectory optimization and scaling methods are costly and stru...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FLARE: A Full-Lifecycle Dense Supervision Paradigm for Long-Horizon Coding Agents via Generative Reward Model

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在长周期软件工程（Long-Horizon SWE）任务中，基于大语言模型（LLM）的智能体通常依赖稀疏的二元奖励信号（Pass/Fail），这导致严重的**信用分配危机（credit assignment crisis）**。当一个代理在执行数十步后失败时，系统无法定位具体哪一步出错，从而浪费大量失败但信息丰富的探索轨迹。此外，现有的测试时扩展（test-time scaling, TTS）方法如全局重采样（Global Rollout）计算成本高昂，而启发式状态复用缺乏因果诊断能力。

### 提出了什么新方法或新思路
本文提出 **FLARE**（Full-Lifecycle Alignment and Reward Engine），一种全新的**全生命周期密集监督范式**，其核心是引入一个轻量级的**生成式奖励模型**（Generative Reward Model, GRM）。该框架包含两个关键组件：

- **RADAR**（Root-cause Attribution and Diagnostic Analysis Refiner）：一个离线因果感知诊断框架，通过**因果链回溯**（causal-chain backtracking）从失败轨迹中提取高保真、无事后偏见（hindsight-free）的监督信号，用于训练 GRM。
- **GRM**：一个可在线运行的生成式奖励模型，能提供实时、细粒度的步骤级风险反馈，输出包括：
  - 风险等级（`safe`, `trivial`, `critical`）
  - 错误分类（error taxonomy）
  - 因果分析与修复建议

FLARE 将 GRM 应用于整个代理生命周期：
- **测试时**：作为“主动支架”（Active Scaffold），在检测到高风险步骤时中断并重新生成后续动作；
- **训练时**：将 GRM 输出用作 SFT 中的数据筛选标准，以及 RL 中的密集奖励信号。

### 相比现有方法的优势
| 维度 | 现有方法（如 SWE-Replay, Satori-SWE） | FLARE |
|------|----------------------------------------|-------|
| 反馈机制 | 延迟的标量评分或启发式分支 | 实时、结构化的生成式诊断 |
| 干预时机 | 失败后重放（post-hoc replay） | 在线动态干预（online intervention） |
| 训练利用 | 丢弃失败轨迹 | 利用失败轨迹构建密集监督 |
| 资源效率 | 高 token 消耗（N=5 全局采样） | 更少 token 下实现更高成功率 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
在四个主流仓库级 SWE 基准上进行评估：
- **SWE-bench Verified**
- **SWE-bench Pro**
- **SWE-bench Multilingual**
- **SWE-Compass**

这些基准涵盖不同难度、编程语言和项目结构的真实 GitHub issue 修复任务。

### 实验设置和评估指标

#### 测试时干预（RQ1）
- **预算协议**：控制每个任务允许的修复分支数 $N$（如 N=1 或 N=5）
- **评估指标**：
  - **Fail-to-Pass (F2P)**：初始失败任务中至少有一个分支通过的比例（Pass@N）
  - **Pass-to-Pass (P2P)**：初始成功任务中平均通过率（Avg@N），衡量干预对成功行为的保留能力
  - **Token 消耗**：仅统计主代理输出 token，不包括 GRM 和输入

#### SFT 数据选择（RQ2）
- 从 18,000 条推理轨迹中选出 3,000 条高质量轨迹用于微调 Qwen3-30B-A3B
- **评估指标**：下游基准的 **resolve rate**（即 pass rate）
- **训练前验证**：使用 ROC-AUC 衡量 GRM 得分对轨迹优劣的排序能力

#### 强化学习（RQ3）
- 对比稀疏奖励 RL 与 FLARE 提供的密集奖励 RL
- **评估指标**：最终 pass rate
- 所有其他超参一致，仅改变奖励信号

### 基线方法对比
| 类型 | 基线方法 |
|------|---------|
| **测试时修复** |  
| - GR (Global Rollout) | 标准 Pass@N，无状态复用 |
| - Blind-BKR | 在高风险点重启，无诊断指导 |
| - RADAR-DBKR | 离线诊断后重放（post-hoc） |
| - GRM-DBKR (Ours) | 在线 GRM 指导干预 |
| **SFT 数据选择** |
| - Random Sampling | 随机选取轨迹 |
| - Rubric-Supervised Critic | 基于规则的批评者模型 |
| - FLARE Process-Aware Selection | 基于 GRM 的过程感知打分 |
| **RL 奖励设计** |
| - Sparse Outcome Reward | 仅使用最终 Pass/Fail |
| - Dense FLARE Reward | 加入 GRM 步骤级密集奖励 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 测试时效率（RQ1）
| 方法 | N=1 F2P (%) | N=5 F2P (%) | Agent Output Tokens (N=1) |
|------|-------------|-------------|----------------------------|
| GR | 7.80 | 13.72 | 22,738 |
| GRM-DBKR (**Ours**) | **14.10** | **19.59** | **12,517** |

- **FLARE (N=1)** 即可超越 **GR (N=5)** 的成功率（14.10% > 13.72%）
- 同时减少约 **5.3× 的 token 消耗**（22.7K → 12.5K）
- 在 P2P 上也表现最佳（88.43% vs GR 的 75.00%），说明不会破坏已有成功路径

#### ✅ SFT 数据选择（RQ2）
| 方法 | Avg. Pass Rate | 相对提升 |
|------|----------------|----------|
| Random Sampling | 28.86 | — |
| Rubric-Supervised Critic | 30.19 | +4.61% |
| **FLARE Process-Aware (Ours)** | **34.38** | **+19.13%** |

- 在所有四个基准上均显著领先，尤其在 SWE-bench Pro 和 SWE-Compass 上提升明显
- 训练前 ROC-AUC 达到 **75.74%**，远高于随机（50.00%）和启发式基线（69.34%）

#### ✅ 强化学习（RQ3）
| 方法 | Avg. Pass Rate | 相对提升 |
|------|----------------|----------|
| Sparse Outcome Reward | 37.77 | — |
| **Dense FLARE Reward (Ours)** | **41.24** | **+9.19%** |

- 在所有四个基准上均有稳定增益：
  - SWE-bench Verified: +2.80 pts
  - SWE-bench Pro: +3.97 pts
  - SWE-bench Multilingual: +2.67 pts
  - SWE-Compass: +4.45 pts

### 消融实验结果（Ablation Study）

#### SFT 成分消融（Table 3）
移除各得分成分后的平均性能下降：
- 移除 **Risk Severity Factor**：↓8.52%
- 移除 **Error Importance Factor**：↓7.18%
- 其他项（action rarity, recurrence, position-aware）也有稳定贡献（↓3–6%）

表明风险严重性和错误重要性是驱动性能的核心因素。

#### 轨迹打分 ROC-AUC 消融（Table 2）
| 移除组件 | ROC-AUC (%) | 下降幅度 |
|--------|------------|---------|
| 完整 FLARE 分数 | 75.74 | — |
| 移除 Risk Severity | 70.12 | ↓5.62 |
| 移除 Error Importance | 71.06 | ↓4.68 |
| 移除其他项 | ~73.x | ↓2–3 |

进一步验证了关键因子的有效性。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **FLARE 实现了新的帕累托前沿（Pareto Frontier）**：
   - 在更少资源下取得更高成功率，打破了“高算力 vs. 低成功率”的传统权衡。
   - **N=1 的 FLARE 胜过 N=5 的全局重采样**，且节省约 5× token。

2. **GRM 支持全生命周期闭环优化**：
   - 同一诊断信号可用于：
     - 测试时：主动干预（Active Scaffold）
     - SFT：高质量轨迹选择
     - RL：构造密集奖励
   - 实现了从失败中学习的真正闭环。

3. **结构化生成式反馈优于标量评分**：
   - GRM 输出包含 severity、taxonomy 和 repair advice，具有更强的可解释性和可操作性。
   - 支持精准定位而非盲目重试。

4. **过程感知监督显著缓解稀疏奖励问题**：
   - 在 SWE 这类长周期交互任务中，传统的 outcome-only reward 极难优化。
   - FLARE 通过 step-level dense reward 提升了 RL 政策学习效率。

### 方法的局限性
1. **受限于基础代理的能力边界**：
   - 若根本性语义理解缺失（如架构级错误），仅靠 GRM 定位也无法修复。
   - GRM 是“支架”而非“替代”。

2. **GRM 推理带来额外开销**：
   - 虽然减少了主代理 token，但 GRM 自身也有推理成本和延迟。

3. **依赖预定义错误分类体系（error taxonomy）**：
   - 当前 taxonomy 可能无法覆盖所有新型错误模式。
   - 需要人工维护和扩展。

4. **离线标注成本较高**：
   - RADAR 的因果链标注需要复杂的双轨流程（反向追溯 + 主动注入），难以完全自动化。

### 未来工作方向
- 构建自适应、可演进的 error taxonomy
- 探索 GRM 与 actor 的联合训练（joint training）
- 将 FLARE 范式推广至非代码领域（如机器人规划、多跳问答）
- 开发更高效的 GRM 架构以降低部署延迟
- 结合多模态观察（如 UI 日志、调试器输出）增强诊断能力

--- 

> **总结一句话**：  
> FLARE 通过构建一个轻量级、可生成诊断的 GRM，在测试时实现高效在线干预，在训练时复用同一信号进行 SFT 和 RL 优化，首次实现了**从失败轨迹中提取高价值监督信号并贯穿代理全生命周期**的统一框架，在性能和效率上全面超越现有方法。

</details>

---

### 14. [When Does Learning Beat Heuristics? A Case Study in Kubernetes Scheduler Score Plugins](https://arxiv.org/abs/2609.22142)

**Authors**: Wang Xuying, Zhibek Sarypbekova  
**Category**: cs.DC  
**Published**: 2026-09-22  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.22142v1  

#### Abstract
Kubernetes scheduler plugins that score candidate nodes are, in production, hand-tuned heuristics. We ask whether a learned scoring function - trained on real placement decisions from a production cluster trace - can match or exceed these heuristics, and if not, why. We implement an external, HTTP-b...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# When Does Learning Beat Heuristics? A Case Study in Kubernetes Scheduler Score Plugins  
**核心结论与实验结果总结**

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文探讨了一个在 **ML-for-Systems** 领域中的根本性问题：  
**在 Kubernetes 调度器的 Score Plugin 场景中，基于机器学习（ML）的模型是否能超越手工设计的启发式规则（heuristics）？**  
具体而言，作者研究了能否通过训练一个学习型评分函数（learned scoring function），在真实生产集群轨迹（production cluster trace）上复现甚至超越现有调度器（如 Fuxi）的实际决策。

该问题具有代表性，因为尽管近年来出现了大量基于 RL 或 GNN 的学习型调度器（如 Decima、DeepRM），但它们的成功往往依赖于端到端强化学习目标，而本文聚焦于更基础的监督学习子任务——即“能否准确预测历史调度选择”。

---

### 提出了什么新方法或新思路

1. **AIScore：可插拔的外部 Score Plugin 架构**  
   设计并实现了一个名为 `AIScore` 的外部 HTTP 接口驱动的 Score Plugin，集成进 `kube-scheduler-simulator`，支持灵活替换评分逻辑，便于实验验证。

2. **面向真实 trace 的监督学习框架**  
   将调度决策建模为 **Learning-to-Rank** 问题：
   - 正样本：实际被选中的节点（label=100）
   - 负样本：同一时间点资源可行且同 failure domain 的其他候选节点（label=20）

3. **关键的数据预处理技术：Sweep-Line Occupancy Reconstruction**  
   提出并实现了基于事件扫描（sweep-line algorithm）的方法，从静态快照中重建每个任务启动时刻的真实节点资源占用状态。这是使离线 trace 可用于训练的前提条件。

4. **强调评估指标对齐的重要性**  
   明确指出：**回归拟合指标（如 R²）不能反映调度质量**，真正重要的是 **Top-1 Ranking Accuracy** —— 即模型是否将最高分赋予真实被选中的机器。

---

### 相比现有方法的优势

| 维度 | 优势说明 |
|------|----------|
| **方法论严谨性** | 区别于多数只报告下游任务性能（如 job completion time）的学习型调度器工作，本研究直接评估模型对历史决策的还原能力，剥离了 RL 控制流的影响，更具可解释性。 |
| **可复现性与开源** | 所有代码、数据管道、实验脚本均公开，极大提升了系统领域 ML 研究的 reproducibility。 |
| **诊断深度** | 不仅比较性能，还深入分析为何 learned model 表现不佳，揭示了“objective mismatch”这一普遍陷阱。 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **Alibaba Cluster Trace v2018** (`alibaba/clusterdata`)
  - 时间跨度：8 天
  - 规模：约 4000 台机器
  - 数据表：`machine_meta`, `machine_usage`, `batch_task`, `batch_instance`
  - 包含真实的 Fuxi 调度器做出的任务放置决策

---

### 实验设置和评估指标

#### 模型架构
| 模型 | 描述 |
|------|------|
| **Random Forest** | 基于 scikit-learn，使用工程化特征（free CPU/memory、failure domain ID、DAG 结构统计等），15050 棵树，max depth=10 |
| **GNN Prototype** | 使用 GraphSAGE 编码 job 的 task-dependency DAG，输出 per-task embedding，再与节点特征拼接后送入 MLP 得分 |

#### 评估指标
| 指标 | 定义 | 是否关键 |
|------|------|---------|
| **R² (Regression Fit)** | 回归任务上的决定系数，衡量对 100/20 标签的拟合程度 | ❌ 误导性 |
| **Top-1 Ranking Accuracy** | 在每组候选节点中，是否给真实被选中的机器打最高分 | ✅ 核心指标 |
| **Synthetic Load-Balancing Benchmark** | 自定义负载下各节点资源利用率标准差，用于 sanity check |
| **Operational Metrics** | HTTP 推理延迟、内存约束下的稳定性（OOM kills）、调度成功率 |

#### 基线方法对比
| 基线 | 描述 |
|------|------|
| **Random baseline** | 随机打分，Top-1 准确率理论值 ~25% |
| **Max free CPU (heuristic)** | 仅按节点剩余 CPU 排序，简单但有效的一行启发式规则 |
| **Default plugins (NodeResourcesFit 等)** | Kubernetes 默认调度策略，在合成测试中作为对照 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 回归性能（R²）——看似积极但具误导性
| 模型 | Train R² | Test R² |
|------|--------|-------|
| Random Forest (Variant D) | 0.0420 | 0.0415 |
| GNN Prototype | 0.021–0.028 | — |

> 尽管 R² 数值低，但在四轮特征工程迭代中呈现单调提升趋势，容易误判为“持续改进”。

#### 真正关键的指标：Top-1 Ranking Accuracy
| 方法 | 数据规模 | Top-1 Accuracy |
|------|----------|----------------|
| Random baseline | n=~2.9M / ~72.7K | 25.2% / 25.3% |
| Max free CPU (heuristic) | — | **74.0% / 84.3%** |
| Random Forest, Variant D | n=~2.9M | 65.4% |
| GNN prototype | n=~72.7K | 65.8% |

> ⚠️ **核心发现：两个 learned model 均显著低于“最大空闲 CPU”这一简单启发式！**

---

### 消融实验结果

#### Random Forest 特征消融（Table 2）
| 变体 | 改动 | Test R² |
|------|------|--------|
| A | 基础资源特征 + 随机负采样 | 0.0249 |
| B | 加入 failure domain ID | 0.0285 |
| C | 负样本限制在同一 failure domain 内 | 0.0327 |
| D | 加入 DAG 衍生特征（in/out-degree, job width, root/leaf） | **0.0415** |

> 特征工程带来 R² 持续提升，但并未转化为 ranking accuracy 上的领先。

#### 特征重要性分析（Variant D）
- `job_width`（父 job 中任务数量）排名第二（importance=0.170），仅次于 `free_CPU`
- 令人意外：一个无需复杂重构的结构化元信息特征，竟优于需 sweep-line 重建的资源特征

#### GNN vs. Random Forest 对比
- GNN 使用的数据量仅为 RF 的 ~1/40，且节点特征极简（仅 plan_cpu, plan_mem）
- 尽管 R² 更低，但 **Top-1 Accuracy 几乎相同（65.4% vs. 65.8%）**
> 表明早期 R² 差异主要由 **data volume 和 feature richness 不对等** 导致，而非架构劣势

---

## 4. 关键结论和发现

### 论文的主要发现

1. ✅ **Learned models 当前未能击败简单启发式**  
   在 Top-1 ranking accuracy 上，Random Forest 和 GNN 均落后于 “rank by free CPU” 启发式达 **8.6–18.5 个百分点**。

2. 🔍 **失败根源是 Objective Mismatch，而非模型能力不足**  
   - 模型使用 **pointwise regression (MSE)** 训练，优化的是标签拟合；
   - 实际部署需要的是 **ranking/ordering** 能力；
   - 这与 Learning-to-Rank 文献中 RankNet、LambdaMART 的经典洞见一致。

3. 🧠 **结构化元信息（如 job width）可能蕴含强信号**  
   `job_width` 成为第二重要的特征，提示未来应优先探索 job-level metadata，而非一味追求精细资源状态重建。

4. 📊 **R² 是系统性误导指标**  
   即使 R² 持续提升，也不能保证调度相关性能改善。**必须从一开始就使用 ranking-aware metric 进行评估**。

5. 💡 **Architecture 比较需控制变量**  
   不同 pipeline 间的数据量、特征丰富度差异会严重混淆架构优劣判断。本文显示 GNN 在极小数据下达到与 RF 相当的 ranking performance，暗示其潜力未被充分释放。

6. 🛡️ **生产鲁棒性表现良好**  
   - HTTP 推理延迟高达 1.5s 时仍无超时（默认 2s timeout）
   - 容器内存压至 120MB（低于 idle footprint）也未发生 OOM kill
   - 性能退化平滑（graceful degradation），适合实际部署

---

### 方法的局限性

| 局限 | 说明 |
|------|------|
| **单 trace 验证** | 仅使用 Alibaba Cluster Trace，未在 Google Cluster Trace 等其他 trace 上验证泛化性 |
| **未重训以验证假设** | 尚未用 pairwise ranking loss 重新训练模型来验证“objective alignment”能否缩小差距 |
| **GNN 设置不对等** | GNN 使用更少数据和更弱特征，限制了公平比较 |
| **缺乏 confidence interval** | 多数实验为单次运行，缺少统计显著性检验 |
| **小规模测试** | 敏感性分析仅在 3 节点、40 Pod 规模进行，未模拟高并发场景 |

---

### 未来工作方向

1. **Retrain with Ranking Loss**  
   使用 pairwise margin loss 或 LambdaRank 类似机制重新训练模型，验证是否能闭合与启发式的差距。

2. **Fair Architecture Comparison**  
   在相同数据量和特征集下重新比较 Random Forest 与 GNN，排除资源偏差影响。

3. **深入分析 job_width 信号来源**  
   分 bucket 分析不同 job width 下的调度模式，指导 reward shaping。

4. **引入 Oracle Baseline**  
   使用组合优化求解器生成 post-hoc 最优解，作为理想参考基准。

5. **扩展动态 workload 和故障注入**  
   引入 wave arrival、node failure 等现实因素，增强仿真真实性。

6. **跨 trace 验证与多随机种子重复**  
   提升结论的统计可信度和泛化能力。

---

> 📌 **最终结论一句话总结**：  
> **当前 learned scoring models 之所以未能打败 heuristics，并非因为架构不行或数据不够，而是因为训练目标（pointwise regression）与部署需求（ranking selection）不匹配 —— 这是一个 methodological pitfall，而非 technical failure。**

</details>

---

### 15. [Mask-Aware Execution for Efficient JEPA Training](https://arxiv.org/abs/2609.22674)

**Authors**: Md Musfiqur Rahman Sanim, Zhihao Shu, Bahram Afsharmanesh, Amirali Mirian, Wei Niu, Gagan Agrawal  
**Category**: cs.DC  
**Published**: 2026-09-22  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.22674v1  

#### Abstract
Joint Embedding Predictive Architectures (JEPAs) are becoming a core representation-learning primitive and a building block for latent world models across vision, video, audio, brain dynamics, and time series. Despite (potential of) wide deployment, current JEPA training pipelines are inefficient: e...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Mask-Aware Execution for Efficient JEPA Training**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前 **JEPA**（Joint Embedding Predictive Architecture）训练流程存在显著效率瓶颈，主要体现在以下三个方面：
- **冗余计算**：每个输入被多个 mask 分割后，独立执行完整的 encoder-predictor 流程，导致共享部分（如 patch embedding、context encoder）被重复计算。
- **内存密集型 token 路由**：每个 mask 需要独立的 `gather` 和 `scatter` 操作，造成不规则内存访问和高带宽消耗。
- **稠密目标编码器执行**：target encoder 对整个 token 网格进行处理，即使只有少数 masked tokens 参与损失计算，浪费大量算力。

这些问题随着 mask 数量增加而线性恶化，严重限制了 GPU 利用率和训练可扩展性。

---

### **提出了什么新方法或新思路**
作者提出 **M-JEPA**（Mask-Aware Execution for JEPA），一种系统级的执行重构架构，其核心思想是：
> **将 mask-independent 计算与 mask-dependent 操作分离，实现共享执行与稀疏化处理**。

具体创新点包括：

| 创新技术 | 描述 |
|--------|------|
| **Shared Context Encoder Execution** | 将 context encoder 中与 mask 无关的部分（如 patch embedding、部分 transformer 层）统一执行一次，避免跨 mask 重复计算。 |
| **Fused Mask Routing with Backward Support** | 设计融合内核（fused kernel），在单次遍历中完成所有 mask 的 token 路由，并支持高效的反向梯度聚合（fused gradient router），减少多次 scatter 开销。 |
| **Sparse Target Encoder Execution** | 仅对所有 mask 的 target token 并集进行编码，大幅减少不必要的前向计算和激活存储。 |
| **Masked Patch Embedding** | 在单 mask 场景下，直接从输入中提取并投影 masked patches，跳过未参与训练的 patch，利用输入稀疏性加速。 |

这些优化均**不改变 JEPA 的学习目标**，完全保留原始训练语义。

---

### **相比现有方法的优势**
- **非模型修改方案**：不同于 SALT、LeJEPA 等通过更改教师网络或正则化策略来提升效率，M-JEPA 是纯系统级优化，适用于所有 JEPA 变体。
- **兼容通用加速技术**：可与 FlashAttention、DeepSpeed 等结合使用，且能进一步放大收益。
- **端到端加速显著**：在多 mask 设置下实现高达 **1.7×** 的 end-to-end 加速，在高稀疏下单 mask 下 patch embedding 加速达 **4.75×**。
- **降低 CUDA 运行时开销**：减少 kernel launch、内存分配和设备同步次数，提高 GPU 利用率。

---

## **2. 核心实验方法和设置**

### **使用的 JEPA 变体与模态**
实验覆盖五种 JEPA 架构，涵盖多种数据模态：
- **V-JEPA** [6]：视频（Vision）
- **EEG-JEPA** [24]：脑电图（EEG）
- **T-JEPA** [50]：表格数据（Tabular）
- **TS-JEPA** [21]：时间序列（Time Series）
- **Brain-JEPA** [16]：脑动态建模（Brain Dynamics）

以 **V-JEPA** 为主要分析对象，因其 spatiotemporal tokenization 和 multi-mask 训练最具挑战性。

---

### **实验设置**
- **硬件平台**：NVIDIA A100 GPU（80GB HBM），CUDA 12.8
- **精度**：BF16
- **batch size**：每轮处理 10 个输入样本
- **mask 数量**：2–10 个 / 输入（标准 JEPA 设置）
- **训练步骤**：平均 100 次迭代（前 20 次为预热）

---

### **评估指标**
| 指标 | 说明 |
|-----|------|
| **End-to-end Training Latency** | 单个训练 step 的总耗时 |
| **Per-component Speedup** | context encoder、predictor、target encoder、patch embedding 各阶段加速比 |
| **Memory Traffic & Utilization** | GPU 内存占用、带宽利用率、idle time |
| **Loss Convergence** | 验证优化是否影响训练稳定性 |
| **Ablation Study** | 分析各组件（SE/OR/SC）对整体加速的贡献 |

---

### **基线方法对比**
- **Baseline**：原始 JEPA 实现（每个 mask 独立执行完整 pipeline）
- **Baseline + FlashAttention**：启用现代注意力优化，确保公平比较
- **Baseline + torch.compile**：测试主流编译器能否自动优化此类结构
- 所有 baseline 均未进行跨 mask 共享或稀疏化处理。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **端到端训练加速**
| 模型 | 2 masks | 10 masks | 备注 |
|------|--------|---------|------|
| **V-JEPA** | ~1.1× | **1.7×** | 最大加速来自共享执行 |
| **EEG-JEPA** | ~1.2× | ~1.6× | 高维时空输入受益明显 |
| **T-JEPA / Brain-JEPA** | 较高加速 | 达 1.6–1.7× | 高稀疏性和重计算负载更受益于稀疏执行 |
| **TS-JEPA** | 较低加速 | ~1.2× | token 间重叠少，共享机会有限 |

> 图 8 显示：**加速比随 mask 数量增长而提升**，因冗余累积效应增强。

#### ✅ **多 GPU 扩展性**
- 在 2-GPU 和 4-GPU 上，M-JEPA 仍保持约 **1.5×** 加速（10 masks），表明优化效果在分布式场景依然有效。

#### ✅ **组件级加速分解（图 11）**
| 组件 | 加速比（10 masks） | 原因 |
|------|------------------|------|
| **Context Encoder** | **1.7×** | 共享 patch embedding 和中间层 |
| **Predictor** | **1.65×** | 融合路由减少多次 gather |
| **Target Encoder** | **1.1× → 1.18×** | union token 数随 mask 增加而扩大，稀疏度下降，但仍稳定受益 |

#### ✅ **Masked Patch Embedding 单独测试**
- 在高稀疏（70%）条件下，**patch embedding 加速达 4.75×**
- 适用于单 mask 或低 mask count 场景，作为补充优化

#### ✅ **消融实验（图 13）**
逐步启用以下优化（V-JEPA, 10 masks）：
| 阶段 | 加速比 | 贡献占比 |
|------|-------|----------|
| **+ Shared Execution (SE)** | 1.38× | **88.7%** |
| **+ Optimized Routing (OR)** | 1.42× | +6.6% |
| **+ Sparse Computation (SC)** | **1.45×** | +4.7% |

> 结论：**共享执行是最大贡献者**，占总提速近 90%。

#### ✅ **运行时开销控制**
- **Routing map 构造时间 < 0.01%**（表 2），可忽略
- **峰值显存略有上升**（表 4）：最多增加 10%，源于共享激活缓存和 union buffer，但在合理范围内

---

## **4. 关键结论和发现**

### **主要发现**
1. **JEPA 训练瓶颈本质是系统问题而非模型问题**  
   冗余计算和内存瓶颈源于执行流程设计，可通过系统级重构解决，无需改动学习目标。

2. **共享执行是最大加速来源**  
   将 mask-independent 计算（如 patch embedding、context encoder 共享层）集中执行一次，消除线性增长的重复开销。

3. **融合内核显著降低内存边界操作成本**  
   fused routing 和 fused backward slicing 改善内存局部性，缓解 gather/scatter 导致的带宽压力。

4. **稀疏执行在特定场景极具潜力**  
   - sparse target encoder 减少无用计算
   - masked patch embedding 在高 sparsity 下带来数量级加速

5. **优化具有广泛泛化能力**  
   在视频、EEG、文本、时间序列等多种模态的 JEPA 变体上均取得一致加速，验证了方法通用性。

---

### **方法的局限性**
- **共享执行增加激活生命周期**：中间张量需保留更久以供多个 mask 使用，略微增加峰值显存（+5–10%）。
- **multi-mask 场景下 masked patch embedding 效益受限**：当多个 mask 的 patch 并集接近全覆盖时，稀疏优势消失。
- **不适用于 single-mask-only 模型**（如 I-JEPA）中的多 mask 特定优化。
- 当前实现依赖手动 kernel 编写（如 Triton），尚未完全自动化。

---

### **未来工作方向**
1. **扩展至 multi-node 低比特训练**：将 fused mask routing 和 sparse execution 推向更大规模集群。
2. **集成至 JEPA-based world model inference**：探索推理阶段的 mask-aware 执行优化。
3. **自动化编译器支持**：开发能自动识别并重构 JEPA 类多分支数据流的 compiler（如基于 torch.compile 或 Triton）。
4. **动态 mask 调度**：根据 token 重叠度动态分组 mask，进一步提升共享效率。

---

> **一句话总结**：  
> M-JEPA 通过 **mask-aware execution** 重构 JEPA 训练流程，在不改变学习目标的前提下，实现了高达 **1.7× 的端到端加速**，证明了**执行结构优化**是高效 JEPA 训练的关键杠杆。

</details>

---

### 16. [PRQuant: Permutation Residual Quantization for Low-Overhead Inference](https://arxiv.org/abs/2609.22106)

**Authors**: Peiran Wang, Anqi Wang, Jiaying Zhao, Huiwen Yang, Zhenyu Ming, Rongqian Wang, Yiwu Yao, Kun Tian, Xin Yao, Gong Zhang, Fan Yang, Zhongyi Huang  
**Category**: cs.LG  
**Published**: 2026-09-22  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.22106v1  

#### Abstract
Accuracy of Low-bit quantization of linear layers is often dominated by a small number of outliers. Although existing methods, such as smoothing, rotation, or residual-based approaches, may mitigate this problem, they often introduce new accuracy bottlenecks to weights. Besides, most of these techni...

---

### 17. [Efficient Reasoning Exploration via State-Conditioned Latent Steering with Progress Guidance](https://arxiv.org/abs/2609.24066)

**Authors**: Hengyuan Zhang, Chenming Shang, Zunhai Su, Xiao Liang, Hui Shen, Jing Xiong, Dawei Li, Shiping Yang, Kailai Yang, Wei Zhang, Ruobing Xie, Hayden Kwok-Hay So, Ngai Wong  
**Category**: cs.CL  
**Published**: 2026-09-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.24066v1  

#### Abstract
Best-of-$N$ is a widely used inference strategy for complex reasoning, whose effectiveness depends on whether sampled candidates can cover diverse and high-quality reasoning paths. However, post-trained reasoning models often suffer from \emph{exploration collapse}, where independent rollouts repeat...

---

### 18. [Conduit: An Experience Data Plane for Distributed Reinforcement Learning](https://arxiv.org/abs/2609.24456)

**Authors**: Sitong Zhang, Tuo Shi, Mario Di Francesco, Zeke Wang, Bo Zhao  
**Category**: cs.DC  
**Published**: 2026-09-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.24456v1  

#### Abstract
Distributed reinforcement learning (RL) scales training by parallelizing actors and learners around an Experience Buffer. As RL workloads grow, however, the buffer becomes more than a replay queue: it is the storage substrate of a large-capacity, latency-critical experience path that every iteration...

---

### 19. [Universal Observatory Graphs for Distributed Sky Coverage and Artificial Intelligence Based Interplanetary Routing](https://arxiv.org/abs/2609.22244)

**Authors**: Mohammed Abdel Razek  
**Category**: cs.LG  
**Published**: 2026-09-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.22244v1  

#### Abstract
This research proposes the Universal Observatory Graph (UOG), an AI-driven framework for distributed astronomical observation across the Solar System. The proposed architecture models autonomous observatories located at the Sun planet L2 Lagrange points as nodes in a weighted graph, while communicat...

---

### 20. [Prioritized Rollouts for Efficient World Model-based Vision-Language-Action Policy Optimization](https://arxiv.org/abs/2609.22879)

**Authors**: Yifei Sheng, Haoxiang Ren, Zhilong Zhang, Haonan Wang, Runjie Xu, Yihao Sun, Nan Tang, Zhichao Wu, Lei Yuan, Haoxin Lin, Yang Yu  
**Category**: cs.LG  
**Published**: 2026-09-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.22879v1  

#### Abstract
Vision-Language-Action (VLA) models have emerged as a powerful paradigm for embodied intelligence, but fine-tuning them with reinforcement learning (RL) remains constrained by the cost of real-world robot interaction. Model-based reinforcement learning (MBRL) reduces this cost by using a learned wor...

---

### 21. [Acceptance-Aware Draft Model Training for Speculative Decoding](https://arxiv.org/abs/2609.24150)

**Authors**: Tianhua Xia, Mugilan Ganesan, Yifei Feng, Haiyu Wang, Maximilian Egger, Sai Qian Zhang  
**Category**: cs.LG  
**Published**: 2026-09-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.24150v1  

#### Abstract
Speculative decoding accelerates large language model (LLM) inference by using a lightweight draft model to generate multiple candidate tokens that are verified by the target model in a single forward pass. Its speedup is largely determined by the acceptance length, yet existing draft-model training...

---

### 22. [Opinion Leader Dynamics: How Sparse Attention Shapes Token Clustering](https://arxiv.org/abs/2609.24202)

**Authors**: Jingkun Liu, Yue Song  
**Category**: cs.LG  
**Published**: 2026-09-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.24202v1  

#### Abstract
Sparse attention reduces the quadratic cost of global self-attention while retaining strong empirical performance, but how its restricted interactions shape the evolution of token representations remains theoretically underexplored. Modeling tokens as particles on the unit sphere, we introduce opini...

---

### 23. [One Prompt Does Not Fit All: Self-Meta-Evolve for Personalized Information Extraction](https://arxiv.org/abs/2609.21626)

**Authors**: Hongliang Li, Lu Wang, Yong Xu, Hanyang Chen, Zhitao Hou, Xiaoting Qin, Song Ge, Qingwei Lin, Dongmei Zhang  
**Category**: cs.AI  
**Published**: 2026-09-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.21626v1  

#### Abstract
Large language models (LLMs) are increasingly deployed for enterprise information extraction (IE), where the same document must be reorganized differently for each user. Existing prompt optimization methods, however, rely on a single prompt optimized against a global objective, which is misaligned w...

---

### 24. [Balancing Reasoning and Hardware Constraints in RAG Pipelines for Ukrainian Multi-Domain Document Understanding](https://arxiv.org/abs/2609.22124)

**Authors**: Illya Havrylov  
**Category**: cs.CL  
**Published**: 2026-09-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.22124v1  

#### Abstract
This paper describes the system submitted to the UNLP 2026 Shared Task on Multi-Domain Document Understanding. The challenge required extracting precise answers, document IDs, and page numbers from a diverse corpus of Ukrainian PDF documents within a strict 9-hour offline Kaggle execution limit. Dur...

---

### 25. [EAVer: Long-Form Factuality Verification as an End-to-End Agentic Policy](https://arxiv.org/abs/2609.22223)

**Authors**: Kening Zheng, Aoying Zheng, Zhigang Chang, Yazhi Guo, Miaotian Guo, Qingwei Zong, Xianhai Xie, Weiqiang Jin, Chengze Li, Hanrong Zhang, Jie Yang, Wei-Chieh Huang, Lingzhe Zhang, Liancheng Fang, Xin Zou, Hanqian Li, Jiahao Huo, Yibo Yan, Zizhuang Deng, Lei Miao, Wei Guo, Haihong Tang, Bo Zheng, Philip S. Yu  
**Category**: cs.CL  
**Published**: 2026-09-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.22223v1  

#### Abstract
Long-form factuality verification is commonly implemented as a static decompose-search-verify pipeline, with separately prompted modules processing claims and invoking external search. Treating claims independently makes LLM and search calls scale with claim count and causes repeated searches for ov...

---

### 26. [Toward Personalized Sleep Guidance from Wearable Data Using Language Models](https://arxiv.org/abs/2609.22463)

**Authors**: Yusheng Tan, Running Zhao, Sofia Angel, Ninghui Hao, Ash Arian, Nikita N. Dulin, Jay Lin, Ou Zhu, Faiza Shaik, Xinxing Yang, Bonnie W. Leung, Katie Roster, Arlene Ruiz de Luzuriaga, Kenneth Lee, Alejandra Lastra, Habibul Ahsan, Guihong Wan  
**Category**: cs.CL  
**Published**: 2026-09-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.22463v1  

#### Abstract
Sleep monitoring using wearable data has shown promise for personal health, yet large language model (LLM)-based summarization and question answering remain insufficient for personalized sleep guidance. Training specialized models, however, often requires costly expert annotation. Moreover, privacy ...

---

### 27. [Efficient LLM Distillation for Bangladesh Legal Context: A Smartphone-Compatible Retrieval-Augmented Generation Model](https://arxiv.org/abs/2609.24177)

**Authors**: MD. Nafis Kamal, Mahadi Hasan Fahim, Talha Ridwan, Nadifa Zaman, Fariha Roushon Florin, Farig Yousuf Sadeque, Saadat Rafid Ahmed  
**Category**: cs.CL  
**Published**: 2026-09-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.24177v1  

#### Abstract
Legal information in Bangladesh is inaccessible to most citizens. Statutory text is English-only, trained lawyers are concentrated in urban centres, and cloud-dependent AI fails where mobile connectivity is unreliable, a setting in which hallucinated legal text causes direct harm. The system address...

---

### 28. [Who Pays for the KV Cache? Attributing Shared AI Inference Spend Across Kubernetes and LLM Provider Bills](https://arxiv.org/abs/2609.24991)

**Authors**: Timothy Urista  
**Category**: cs.DC  
**Published**: 2026-09-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.24991v1  

#### Abstract
Organizations pay for AI through disconnected ledgers: Kubernetes allocations for self-hosted inference, gateway logs, and per-token bills from API providers. We present unalloc, an open-source tool that joins OpenCost, LiteLLM, OpenAI and Anthropic cost data into one exact ledger and reports the sh...

---

### 29. [SCALE: Simulation-Calibrated Amortized Learning for Energy Materials (A hybrid architecture connecting deterministic modeling, real-world data, and transformer-scale inference for accelerated energy-materials discovery)](https://arxiv.org/abs/2609.22233)

**Authors**: Kuan Huang, Bo Bai  
**Category**: cs.LG  
**Published**: 2026-09-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.22233v1  

#### Abstract
Energy systems face converging pressures for security, affordability, resilience, and sustainability, creating a need for faster discovery of deployable energy materials. Here we introduce SCALE (Simulation-Calibrated Amortized Learning for Energy Materials), a physics-grounded, real-world-data-cali...

---

### 30. [FlashBoB: I/O-Efficient Exact Backward-over-Backward for Softmax Attention](https://arxiv.org/abs/2609.24089)

**Authors**: Anthony Givans, Michael Crawshaw, Mingrui Liu  
**Category**: cs.LG  
**Published**: 2026-09-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.24089v1  

#### Abstract
Transformer models built on the attention mechanism have become a central building block in modern deep learning, yet softmax attention remains a major bottleneck for long-context workloads. While FlashAttention makes the forward and first backward passes I/O-efficient, it does not support backward-...

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

# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-07-10 08:55:45 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [On the Limitations of Non-GPU AI Accelerators for Large-Model Inference: A Field Study of MoE and Multimodal Serving on Huawei Ascend](https://arxiv.org/abs/2607.08215)

**Authors**: Zheng Yu  
**Category**: cs.DC  
**Published**: 2026-07-10  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2607.08215v1  

#### Abstract
Non-GPU AI accelerators are increasingly adopted as alternatives to general-purpose GPUs for large-model inference, but the real engineering cost of migrating demanding workloads beyond CUDA remains poorly documented. We present a field study of deploying two large inference workloads on a 16-device...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：On the Limitations of Non-GPU AI Accelerators for Large-Model Inference: A Field Study of MoE and Multimodal Serving on Huawei Ascend

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文系统性地研究了在非GPU AI加速器（特别是华为Ascend 910）上部署大规模语言模型（尤其是MoE和多模态模型）进行推理时所面临的实际挑战。尽管Ascend等替代性加速器因供应、成本和可用性优势被广泛采用，但从成熟的CUDA生态迁移至此类平台的实际工程代价缺乏公开记录。

论文填补了这一空白，通过真实场景下的**现场研究**（field study），揭示了从软件栈到硬件层面的一系列未被充分文档化的限制。

### 提出了什么新方法或新思路
- **提出八类平台级限制分类框架**：将观察到的问题归纳为八个系统性的类别，形成可复用的分析模板：
  1. 软件栈成熟度与算子/功能覆盖缺口（Software-stack maturity and operator/feature coverage gaps）
  2. 多轴并行脆弱性（Fragile multi-axis parallelism）
  3. 内核级数值与稳定性故障（Kernel-level numerical/stability faults）
  4. 不成熟的图编译（Immature graph compilation）
  5. 高级功能缺失（Advanced-feature gaps）
  6. 性能与扩展性瓶颈（Performance and scalability ceilings）
  7. 运维可靠性与可观测性弱（Operational reliability and observability）
  8. 生态碎片化与移植税（Ecosystem fragmentation and portability tax）

- **提供通用、厂商无关的迁移策略**：基于经验提炼出一套适用于任何非GPU加速器的通用采纳策略，如差分测试、CI集成、容错服务设计等。

### 相比现有方法的优势
- **真实性高**：不同于理论分析或简化基准测试，本研究基于两个复杂且真实的推理流水线（LLM-as-a-judge 和 医疗视觉语言模型），具有强实践指导意义。
- **可复现性强**：详细记录了错误码、环境变量、补丁代码、日志片段等低层细节，便于其他团队复现和规避同类问题。
- **普适性强**：虽然实验对象是Ascend，但其识别的根本原因（如编译器不成熟、特征组合覆盖不足、生态分裂）对其他NPU平台也具参考价值。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **Case Study A（LLM-as-a-judge）**：
  - 输入：来自20个前沿LLM（如GPT-5.1、Gemini 3 Pro、Claude Haiku 4.5等）在价值观对齐与安全相关问题上的响应。
  - 规模：每个模型处理数万个prompt。
  - 输出：由DeepSeek-V4-Flash-w8a8-mtp模型打分并生成JSON格式判决。

- **Case Study B（多模态医疗VLM）**：
  - 数据集：**MMMU** 和 **MMMU-Pro**，用于评估视觉语言理解能力。
  - 模型：DeepSeek-V4-Flash-Vision，融合冻结的Qwen3.5视觉塔与DeepSeek-V4-Flash MoE解码器，通过训练好的merger/bridge连接。

### 实验设置
- **硬件平台**：单节点16卡Huawei Ascend 910。
- **软件栈**：
  - CANN（Compute Architecture for Neural Networks）工具包
  - vLLM-Ascend 推理引擎（0.13系列版本）
  - PyTorch + `torch_npu` 后端
- **部署方式**：
  - 使用OpenAI兼容接口暴露服务。
  - 应用了12个源码级补丁以修复初始化和运行时缺陷。
  - 对部分高级功能（如sequence parallelism、fused MC2、MTP speculative decoding）主动禁用以保证正确性。

### 评估指标
- **功能性验证**：
  - 最终benchmark得分是否匹配原生框架结果（用于确认数值正确性）。
- **性能行为观测**：
  - 并发请求下的aggregate throughput与queue depth变化。
  - per-stream throughput随并发增长的变化趋势。
  - 冷启动时间（cold-start time）。
- **稳定性监控**：
  - 设备异常码捕获（如507015、507035）。
  - 日志中共享内存阻塞、内核崩溃等信号。

### 基线方法对比
- **隐含基线**：CUDA生态下的标准vLLM实现（作为“黄金标准”）。
- **对比维度**：
  - 功能完整性（feature parity）
  - 数值一致性（numerical correctness）
  - 吞吐量与并发表现
  - 可靠性与运维成本

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | 结果 |
|------|------|
| **Case Study A 内存占用** | W8A8量化模型约300 GB |
| **Case Study B 内存占用** | bf16精度模型约540 GB，每卡平均使用~59 GB（接近64 GB上限） |
| **并发“甜点”（sweet spot）** | 约 **4个并发流**时达到最高吞吐；超过后队列饱和，总吞吐下降（见Figure 5） |
| **冷启动时间** | 数分钟（需完成算子编译与权重/KV缓存量化） |
| **MMMU-Pro 准确率** | **48.0%**（forced-direct协议下） |
| **MMMU 准确率** | **57.8%**（validation set） |

> ✅ **关键结论**：上述分数与原生框架参考值持平甚至略优 → 表明最终部署实现了**端到端数值正确性**。

### 与基线方法的对比结果
- **功能差距显著**：
  - 需手动打12个补丁才能启动。
  - 多项优化功能（FlashComm、fused MC2、MTP）必须关闭以避免崩溃。
- **性能不可预测**：
  - 吞吐非单调增长，存在明显“过载即崩”现象。
  - 缺乏JIT编译支持（无TorchInductor），导致无法动态融合kernel。
- **稳定性差**：
  - 出现aicore/vector-core执行异常（runtime code 507015/507035），需外部watchdog自动重启。
  - 图编译路径处于实验阶段，有OOM和hang风险。

### 消融实验结果（间接体现）
虽然未明确列出消融表，但文中多次通过启用/禁用特定功能来验证影响：
- **开启FlashComm → 张量尺寸不匹配错误**（sharded vs full-size inputs_embeds）
- **启用fused MC2 → 内存预估阶段token_ids缺失导致失败**
- **启用full-decode图捕获 → 存在OOM和推理挂起警告**
- **启用MTP speculative decoding → 增加不稳定性和调度复杂度**

这些尝试本质上构成了**功能级消融实验**，证明了“简化配置”是保障稳定性的必要手段。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Ascend平台可以运行大型MoE和多模态模型**，但前提是投入大量工程努力进行修补和调优。
2. **真正的成本不在硬件采购，而在工程整合**：迁移的TCO（Total Cost of Ownership）主要由以下构成：
   - 12个源码补丁维护
   - 功能降级带来的吞吐损失
   - 外部容错机制开发（watchdog、log scraping）
   - 版本锁定与配置管理
3. **平台级限制具有共性根源**：
   - 编译器与算子库不成熟
   - 特征组合测试覆盖率极低（如PP > 1 + MoE + multimodal）
   - 缺乏生产级可靠性和可观测性设计
   - 生态碎片化导致“移植税”高昂

### 方法的局限性
- **研究范围有限**：
  - 单一厂商（Huawei Ascend）、单一引擎版本（vLLM-Ascend 0.13.x）、单节点部署。
  - 仅针对两种高度复杂的模型架构（MoE + 多模态），不反映简单dense模型的表现。
- **非标准化评测**：
  - 性能数据为粗粒度操作观察，非严格控制变量的benchmark。
  - 绝对数值依赖于请求混合模式和生成长度，不具备直接横向比较意义。

### 未来工作方向
1. **推动生态系统改进**：
   - 要求厂商提供更稳健的默认行为（graceful fallback而非硬崩溃）
   - 改善诊断信息层级化（将设备错误映射回具体operator或layer）
   - 加强feature cross-product的测试覆盖（如MoE × PP × multimodal）
   - 发布清晰的version/feature兼容矩阵

2. **构建更健壮的服务架构**：
   - 默认启用fault-tolerant serving（健康检查 + 自动重启 + 请求重试）
   - 在CI中持续运行target stack的真实decode路径
   - 构建thin portability layer隔离vendor-specific logic

3. **社区协作降低重复成本**：
   - 共享补丁、复现案例和最佳实践
   - 向上游贡献fixes以缩小与主干分支的差距

---

> 📌 **总结一句话**：  
> **Ascend等非GPU加速器具备运行大模型的能力，但当前软件栈的不成熟使其工程落地成本极高；成功的迁移不仅依赖技术适配，更需要系统性的工程策略来应对可靠性、可维护性和生态碎片化挑战。**

</details>

---

### 2. [MobiDiff: Semantic-Aware Multi-Channel Discrete Diffusion for Human Mobility Data Generation](https://arxiv.org/abs/2607.08357)

**Authors**: Rongchao Xu, Lin Jiang, Dahai Yu, Ximiao Li, Taichi Liu, Desheng Zhang, Yuan Tian, Guang Wang  
**Category**: cs.AI  
**Published**: 2026-07-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2607.08357v1  

#### Abstract
Human mobility data are essential for transportation optimization, urban planning, and resource allocation, yet real-world mobility data are costly to collect and difficult to share due to privacy concerns. Recent diffusion-based methods have shown promise in synthesizing realistic mobility patterns...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《MobiDiff: Semantic-Aware Multi-Channel Discrete Diffusion for Human Mobility Data Generation》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有基于 diffusion 的 human mobility 数据生成方法通常依赖于连续或潜在的时空轨迹表示（如连续坐标、latent trace），并需要复杂的插值、粗到精（coarse-to-fine）生成流程。这些方法难以直接建模具有显式区域、活动、时间和间隔结构的**离散语义事件**，导致生成过程与人类移动行为的离散本质不一致。

此外，这类方法在效率、可解释性和对时间结构的建模上存在不足。

### 🚀 提出的新方法：MobiDiff
作者提出 **MobiDiff** ——一种端到端的、语义感知的多通道离散扩散框架，用于大规模 human mobility 数据生成。

#### 核心创新点：
- **将 mobility 轨迹建模为多通道语义骨架（multi-channel semantic skeleton）序列**  
  每个 check-in 事件被分解为五个离散通道：
  - 宏观区域（macro-region）
  - 微观区域（micro-region）
  - 活动类别（activity category）
  - 绝对时间（absolute-time）
  - 时间间隔（gap-time）

- **引入结构化掩码策略（structured masking）**  
  在三个粒度上进行掩码以联合学习：
  - **Event-level masking**：整个事件的所有通道被掩码 → 学习轨迹级上下文依赖
  - **Group-level masking**：空间组 `{M, m}`、时间组 `{a, g}` 或混合组 `{M,m,c}`, `{c,a,g}` → 学习跨因素一致性
  - **Channel-level masking**：单个通道被掩码 → 学习事件内部一致性

- **数值感知嵌入（numeric-aware embeddings）**  
  尽管使用离散 token，但保留其隐含的数值意义：
  - 区域 token 关联地理坐标（经纬度）
  - 时间 token 编码周期性（sin/cos 特征）
  - 间隔时间 token 使用 log 变换后的中心值
  → 通过轻量 MLP 投影增强 token embedding

- **端到端离散扩散，避免复杂 pipeline**  
  不再需要 latent trace 构造、插值或 coarse-to-fine 分阶段生成，直接在离散语义空间中完成去噪。

### 🔍 相比现有方法的优势
| 方面 | MobiDiff 优势 |
|------|---------------|
| **建模范式** | 直接在离散语义空间操作，更符合 mobility 数据本质 |
| **效率** | 推理速度快，无需多阶段采样 |
| **可解释性** | 生成的是语义骨架，便于分析和控制 |
| **时间建模** | 显式建模绝对时间和间隔时间，显著提升时间分布保真度 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
在三个真实世界城市级 human mobility 数据集上进行评估：

| 城市 | 用户数 | POI 数 | Check-in 数 | 轨迹数 | 平均长度 |
|------|--------|--------|-------------|--------|----------|
| Atlanta | 174,787 | 5,178 | 2,324,746 | 289,175 | 8.04 |
| Boston | 72,296 | 2,843 | 1,075,084 | 126,366 | 8.51 |
| Seattle | 113,509 | 4,850 | 1,871,657 | 215,128 | 8.70 |

所有数据转换为统一的 multi-channel semantic skeleton 表示。

### ⚙️ 实验设置
- **训练/测试划分**：模型在训练集上训练，在独立测试集上评估。
- **输出格式对齐**：生成结果统一转换为与真实轨迹相同的格式，使用相同特征提取 pipeline。
- **评估视角**：
  - **保真度（Fidelity）**
  - **实用性（Utility）**
  - **经验暴露风险（Empirical Exposure Risk）**

### 📈 评估指标

#### （1）保真度（Fidelity）
采用 **Jensen-Shannon Divergence (JSD)** 衡量生成与真实数据分布之间的相似性：
- **空间组**：travel distance, movement radius
- **时间组**：inter-event interval, trajectory length, duration
- **语义组**：POI diversity, POI entropy, category diversity, category transition
- **总体指标**：所有特征 JSD 的平均值（越低越好）

#### （2）实用性（Utility）
下游任务：next-event prediction
- **Setting 1**: 使用合成数据完全替代真实训练数据（synthetic replacement）
- **Setting 2**: 合成数据作为小样本增强（low-data augmentation）
- **指标**：normalized Top-k ratio（越高越好），相对于全量真实数据训练的性能归一化

#### （3）经验暴露风险（Empirical Exposure Risk）
- **tight nearest-training overlap**：比较每条生成轨迹与其最近训练轨迹的事件匹配程度
- 空间容忍 ≤ 0.2km，时间容忍 ≤ 30分钟
- 指标：P95 overlap, mean overlap, calibration ratio（越低表示记忆化越少）

### 🆚 基线方法对比
| 方法 | 类型 | 简介 |
|------|------|------|
| **GeoGen [14]** | Two-stage diffusion | 粗到精框架，先生成宏观再细化微观位置 |
| **SynHAT [16]** | Two-stage diffusion | 针对 human activity trace 设计的 coarse-to-fine 扩散框架 |
| **MoveSim [3]** | GAN-based simulator | 基于对抗学习的 mobility 模拟器，非扩散类强基线 |

---

## 3. 主要实验结果和性能指标

### 📉 RQ1: Generation Fidelity（保真度）

| 方法 | Avg. JSD（三城平均） | Temporal JSD（interval/length/duration） | Semantic JSD（POI div./entropy） |
|------|------------------------|-----------------------------------------|-------------------------------|
| MobiDiff | **0.084** | **0.039** | **0.107** |
| GeoGen | 0.216 | 0.290 | 0.251 |
| SynHAT | 0.174 | 0.263 | 0.181 |
| MoveSim | 0.161 | 0.255 | 0.142 |

- **MobiDiff 在时间结构建模上表现最优**，JSD 下降约 **60–87%**
- **语义覆盖（POI 多样性与熵）也显著优于其他方法**
- **空间分布（distance/radius）仍有改进空间**，尤其在 Boston 上表现较弱

> ✅ 支持核心设计：显式建模 `absolute-time` 和 `gap-time` 通道极大提升了时间保真度。

### ⏱️ RQ2: Inference Efficiency（推理效率）

| 方法 | 平均吞吐量（trajectories/sec） | 相对速度提升 | 峰值 GPU 内存（MB） |
|------|------------------------------|--------------|--------------------|
| MobiDiff | **19.82** | **5.3× > GeoGen**, **1.9× > SynHAT** | 1168 |
| GeoGen | 3.75 | — | 346 |
| SynHAT | 10.22 | — | 332 |
| MoveSim | 33.41 | 最快 | 744 |

- MobiDiff 是 **最快 diffusion-based 方法**，因采用单阶段离散采样，避免了 coarse-to-fine 流程
- 但 **MoveSim 更快且内存更低**，说明非扩散方法仍具效率优势
- MobiDiff 内存占用较高，可能源于双向 Transformer 和多通道表示

### 🧪 RQ3: Downstream Utility（下游实用性）

#### （1）Synthetic Replacement Setting

| 方法 | Macro Top-5 | POI Top-20 | Cat. Top-5 |
|------|-------------|------------|------------|
| MobiDiff | 0.654 | **0.480** | 0.880 |
| GeoGen | 0.347 | 0.142 | 0.634 |
| SynHAT | 0.920 | 0.275 | 0.886 |
| MoveSim | 0.875 | **0.666** | 0.954 |

- MobiDiff 的 POI 替代效用是 GeoGen 的 **~5倍**，SynHAT 的 **~1.9倍**
- 但仍落后于 MoveSim，表明 autoregressive/simulation 式训练更利于 next-event prediction

#### （2）Low-data Augmentation Setting

| 方法 | Macro Top-5 | POI Top-20 | Cat. Top-5 |
|------|-------------|------------|------------|
| MobiDiff | 0.907 | **0.665** | 0.962 |
| GeoGen | 0.841 | 0.559 | 0.930 |
| SynHAT | 0.949 | 0.471 | 0.928 |
| MoveSim | 0.934 | **0.742** | 0.961 |

- MobiDiff 在 POI 增强方面表现优异，接近 MoveSim
- 表明生成的语义骨架能有效补充真实数据中的稀疏信号

> ❗ 局限性：masked denoising 的目标是重建缺失事件，而 downstream task 是预测未来事件，两者目标不一致。

### 🔐 RQ4: Empirical Exposure Risk（暴露风险）

| 方法 | 平均 P95 Overlap | 平均 Mean Overlap |
|------|------------------|-------------------|
| MobiDiff | 0.783 | 0.660 |
| MoveSim | 1.000 | 0.786 |
| GeoGen | 0.686 | **0.437** |
| SynHAT | 0.771 | **0.463** |

- MobiDiff 暴露风险低于 MoveSim，表明其未过度记忆训练轨迹
- 但 **高于 GeoGen 和 SynHAT**，说明 diffusion-based 方法在隐私保护上仍有潜力

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **离散扩散适用于 human mobility 生成**  
   MobiDiff 验证了直接在离散语义空间中进行 diffusion 是可行且高效的路径。

2. **时间结构建模取得突破性进展**  
   显式建模 `absolute-time` 和 `gap-time` 通道使 MobiDiff 在 interval、length、duration 分布上远超现有方法。

3. **端到端架构带来显著效率优势**  
   相比 coarse-to-fine diffusion 方法（如 GeoGen、SynHAT），MobiDiff 推理速度快 **5.3倍以上**。

4. **语义骨架有助于实用性和可控性**  
   生成的 multi-channel events 可用于下游任务增强，尤其在低资源场景下效果明显。

5. **暴露风险低于强非扩散基线**  
   相比 MoveSim，MobiDiff 的训练轨迹重叠率更低，表明其泛化能力更强。

### ⚠️ 方法的局限性
- **空间保真度有待提升**：在 distance 和 radius 分布上不如 SynHAT 和 MoveSim，尤其是在 Boston 表现较差。
- **实用性受限于训练目标错配**：masked reconstruction 与 next-event prediction 目标不一致，影响推荐任务表现。
- **内存消耗高**：峰值 GPU 占用达 1168MB，高于所有 baseline。
- **未提供 formal privacy guarantee**：仅通过经验指标评估暴露风险，非差分隐私级别保障。

### 🔮 未来工作方向
1. **提升空间建模能力**：结合图神经网络或 spatial hierarchy 建模区域关系。
2. **设计 utility-aware objective**：在训练中引入 downstream task 对齐的目标函数。
3. **加强隐私机制**：集成 differential privacy 或 contrastive regularization 减少训练数据暴露。
4. **探索更高效架构**：优化多通道表示与 denoising 流程，降低内存与计算开销。
5. **扩展至更多应用场景**：如 disaster response、healthcare visit prediction 等领域。

---

> 💡 **总结一句话**：  
> **MobiDiff 开辟了一条“语义优先、离散去噪”的 human mobility 生成新范式，在时间保真度和推理效率上显著超越主流 diffusion 方法，为可解释、高效、结构化的 synthetic mobility data generation 提供了重要方向。**

</details>

---

### 3. [Hidden Decoding at Scale: Latent Computation Scaling for Large Language Models](https://arxiv.org/abs/2607.08186)

**Authors**: Aiwei Liu, Cheng Shi, Chuhan Wu, Ci Lei, Di Lu, Donald He, Fan Zhang, Fanhao Kong, Feifei Zhang, Guan Wang, Haicheng Wang, Haoyu Liu, Houjin Yu, Jiachen Ding, Jiayi Feng, Jie Zhou, Jijun Chi, Jindi Shi, Jing Lei, Junjie Zhang, Laiyi Li, Le Tian, Linhao Zhang, Miao Fan, Sijun Zhang, Wei Jia, Weiwei Shi, Wenhan Li, Wentao Zhao, Wenteng Liang, Xiao Zhou, Xiaojin Zhou, Xihuai Wang, Xinyu Gao, Xuanliang Wang, Xuyang Ao, Yang Yu, Yangxiu You, Yinuo Zhao, Yufei Kuang, Yufei Wang, Yuan Liu, Yuan Liu, Yuwen Chen, Zhencong Tian, Zhongyin Zhao, Zilin Yu, Zitao Wang  
**Category**: cs.CL  
**Published**: 2026-07-10  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.08186v1  

#### Abstract
Scaling Large Language Models (LLMs) has been driven mainly by enlarging the Transformer backbone, but for an already-strong model this requires another round of costly pretraining. We study whether an existing backbone can keep improving by allocating more computation to each token while leaving th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Hidden Decoding at Scale: Latent Computation Scaling for Large Language Models**

---

## **1. 主要贡献和创新点**

### **解决的问题**
当前大规模语言模型（Large Language Models, LLMs）的扩展主要依赖于增大 Transformer 骨干网络（backbone），但这需要昂贵的重新预训练，成本高昂。本文提出了一种**在不改变骨干网络的前提下，通过增加每个 token 的内部计算量来持续提升模型能力**的新路径。

现有方法如深度循环（depth-recurrent 或 looped Transformers）虽然能增加每 token 计算量，但其循环执行模式与用于训练超大模型的 **pipeline parallelism** 不兼容，难以扩展到百亿参数以上规模。

---

### **提出的新方法：Hidden Decoding**
作者提出了 **Hidden Decoding (HD)** ——一种基于序列长度扩展的 latent computation scaling 方法，应用于 **Continued Pretraining (CPT)** 过程中。

#### **核心机制**
- 将每个输入 token 扩展为 `n` 个独立流（streams），每个流有独立的 embedding 表（E₁, ..., Eₙ）。
- 中间流（前 `n-1` 个）作为“隐状态”进行内部计算，仅最后一个流负责预测下一个 token。
- 保留中间流的 Key-Value (KV) 缓存，使其计算结果可作为上下文供后续 token 使用。

#### **关键技术：Stream-Factorized Attention**
为降低扩展后注意力计算的复杂度（从 O(n²L²) 下降到近似线性），提出：
- 大多数层只在各自 stream 内部进行 attention（intra-stream）；
- 只有少数层允许跨 stream 注意力（cross-stream），复用原模型的 attention 结构（如滑动窗口或全连接）。
- 显著降低了训练开销，使方法可扩展至 MoE 超大规模模型。

---

### **相比现有方法的优势**
| 维度 | Hidden Decoding | Looped Transformers | 其他长度扩展方法 |
|------|------------------|---------------------|------------------|
| **与 pipeline parallelism 兼容性** | ✅ 完全兼容（单次前向传播） | ❌ 不兼容（多次循环阻塞流水线） | ✅ 兼容 |
| **计算效率** | ✅ 近线性增长（O(nL²)） | ❌ 串行叠加，延迟高 | ❌ 密集 attention 成本高（O(n²L²)） |
| **可扩展性** | ✅ 已验证至 617B MoE 模型 | ❌ 最大约 40B，难扩展 | ❌ 未达百B级 MoE |
| **工程实现友好性** | ✅ 输入变长即可，无需修改训练框架 | ❌ 需定制调度逻辑 | ⚠️ 高内存占用 |

> ✅ **首次在 >100B 参数 MoE 模型上成功验证的序列长度扩展方法**

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **主评估套件（Early SFT 后）**：
  - 数学与科学：`GPQA Diamond`, `FrontierMath`, `MathArena Apex`, `HMMT`, `PolyMath`, `PHYBench`, `CritPt`
  - 知识与推理：`MMMLU`, `HLE`, `AA-OmniScience`, `ARC-AGI-2`
  - 编码：`SciCode`, `LiveCodeBench`
  - 指令跟随与代理任务：`IF-Bench`, `Terminal-Bench 2`, `GDPval`

- **消融与缩放研究套件**：
  - `MMLU`, `MMLU-Pro`, `C-Eval`, `SuperGPQA`, `BBH`, `GSM8K`, `MATH`, `SimpleQA`, `HumanEval+`, `MBPP+` 等标准基准

---

### **实验设置**
| 项目 | 设置说明 |
|------|----------|
| **模型架构** | 基于 WeLM MoE 架构（80B 和 617B 总参，激活参数分别为 3B 和 23B） |
| **Hidden Decoding 扩展因子 n** | 主要使用 `n=4`；在 Qwen3-8B 上测试 `n ∈ {2,4,8}` |
| **训练方式** | 在已有强 checkpoint 上进行 **Continued Pretraining (CPT)**，引入 HD |
| **注意力配置** | 多数层 intra-stream，少量层 cross-stream（6 层全连接或滑窗） |
| **上下文长度** | 80B: 256k → 1M；617B: 32k → 128k |
| **评估阶段** | 使用相同的轻量级 **SFT-only post-training**（无 RL），确保公平比较 |

---

### **基线方法对比**
- **Non-HD Baseline**：相同 backbone、训练流程、数据和 post-training，仅关闭 Hidden Decoding。
- **外部参考模型**：Kimi K2.6 (1T total, 32B activated)

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（WeLM-HD4 vs Non-HD）**

#### **WeLM-HD4-80B (n=4) vs WeLM-80B**
| Benchmark | WeLM-80B | WeLM-HD4-80B | Δ |
|---------|--------|-------------|----|
| SciCode | 45.8 | 50.0 | **+4.2** |
| PHYBench | 69.8 | 73.8 | **+4.0** |
| FrontierMath | 45.8 | 49.0 | **+3.2** |
| MMMLU | 84.4 | 85.6 | +1.2 |
| GPQA Diamond | 87.6 | 88.8 | +1.2 |

> ✅ **所有 9 项共享基准均提升**

#### **WeLM-HD4-617B (n=4) vs WeLM-617B**
| Benchmark | WeLM-617B | WeLM-HD4-617B | Δ |
|---------|----------|---------------|----|
| GPQA Diamond | 89.1 | 91.2 | **+2.1** |
| HLE | 33.6 | 35.4 | **+1.8** |
| FrontierMath | 49.0 | 51.0 | **+2.0** |
| MMMLU | 86.4 | 87.5 | +1.1 |
| PHYBench | 75.3 | 76.3 | +1.0 |

> ✅ **同样全面超越基线，在硬核推理任务上增益显著**

#### **扩展因子 n 的缩放效果（Qwen3-8B）**
随着 `n` 增加，性能单调上升：
- MMLU: 79.8 (n=1) → 80.9 (n=2) → 81.9 (n=4) → **82.2 (n=8)**
- BBH: 78.8 → 81.3 → 83.0 → **83.9**
- MATH: 56.0 → 58.2 → 60.0 → **61.1**
- HellaSwag: 79.7 → 83.1 → 85.0 → **85.3**

> 🔁 **证明 `n` 是有效的 scaling knob**

---

### **消融实验结果**

#### **(1) Supervision Design Ablation (6B MoE)**
| 方法 | MMLU | ARC-C | MATH | LM Loss ↓ |
|------|------|-------|------|----------|
| All-token loss (n=2) | 55.5 | 65.8 | 49.6 | 1.880 |
| Sum outputs (n=2) | 55.6 | 71.9 | 49.3 | 1.877 |
| **Hidden Decoding (n=2)** | **56.5** | **74.3** | **50.3** | **1.874** |
| HD (n=3) | 58.5 | 75.7 | 50.3 | 1.857 |

> ✅ **仅监督 final stream 效果最好，中间流应作为 latent computation 状态**

#### **(2) Stream-Factorized Attention Ablation (21B MoE)**
| Full Cross-Stream Layers | Avg Score |
|--------------------------|---------|
| 0 (only local/intra) | 55.88 |
| 1 | 55.88 |
| 4 | **56.49** |
| 27 (all-full) | 56.95 |

> ✅ **只需少数 full cross-stream 层即可接近最优性能，验证了计算效率设计的有效性**

#### **(3) KV Retention Ablation**
- 使用共享 KV（类似 PHD）会降低平均准确率（64.23 → 63.46）
- 证明保留 per-stream KV 对性能有正向作用

---

### **训练与推理成本测量**

| 指标 | 结果 |
|------|------|
| **训练成本（相对 baseline）** |  
| WeLM-HD4-80B (n=4) | **5.1×**（远低于 16× 密集 attention） |
| WeLM-HD4-617B (n=4) | **4.4×** |
| **推理吞吐（batch=1）** | 保持 baseline 的 **83–88%** |
| **高 batch 推理（batch=32, 32k input）** | 下降至 **27%** |

> ⚠️ 推理成本随 batch size 和 context length 增加而显著上升，但在小批量场景下仍实用

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **Hidden Decoding 是首个在 >100B MoE 规模上成功的序列长度扩展方法**，可在不改变 backbone 的情况下持续提升模型能力。
2. ✅ **扩展因子 `n` 是一个有效的 scaling knob**：增大 `n` 能系统性地改善语言建模损失和下游任务表现。
3. ✅ **Stream-Factorized Attention 实现了近线性计算扩展**，是该方法可扩展的关键。
4. ✅ **中间流作为 latent computation 状态有效**：它们形成不同的内部表示，并被最终流读取以生成更优输出。
5. ✅ **KV 缓存分离至关重要**：保留各 stream 的独立 KV 显著影响最终性能。

---

### **方法的局限性**
- **推理吞吐下降明显**：尤其在大批量、长上下文场景下，decode throughput 可降至 27%。
- **embedding 参数增长快**：虽不影响计算主体，但存储需求上升（如 617B 模型从 26.9B → 107.4B embedding 参数）。
- **需 careful 初始化策略**：采用 cyclic replication initialization 来稳定训练。
- **目前仅验证于 WeLM 自研架构**，通用性有待进一步验证。

---

### **未来工作方向**
- 探索更高效的 cross-stream communication 方式（如稀疏混合、动态路由）。
- 优化推理时的 KV 管理，结合 **KV-cache compression** 或 selective caching。
- 将 Hidden Decoding 与其他 scaling 方法（如 test-time compute, MoE 扩展）结合。
- 探索在 vision-language 或 multimodal 模型中的应用潜力。

---

> 🏁 **总结一句话**：  
> **Hidden Decoding 提供了一条工程可行、效果显著的“固定 backbone + 增加 latent computation”的新 scaling 路径，打破了传统必须扩大模型参数才能提升性能的范式，为 LLM 的持续进化开辟了新方向。**

</details>

---

### 4. [BiSCo-LLM: Lookup-Free Binary Spherical Coding for Extreme Low-Bit Large Language Model Compression](https://arxiv.org/abs/2607.08643)

**Authors**: Yuantian Shao, Peisong Wang, Zhilei Liu, Chuangyi Li, Yuanteng Chen, Pengcheng Xie, Yiwu Yao, Zhihui Wei, Jian Cheng  
**Category**: cs.LG  
**Published**: 2026-07-10  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.08643v1  

#### Abstract
Large language models (LLMs) are increasingly constrained by memory capacity, weight bandwidth, and checkpoint storage during deployment. Existing low-bit compression methods mainly follow two directions. Scalar or group-wise quantization is simple and compatible with efficient low-precision kernels...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# BiSCo-LLM: Lookup-Free Binary Spherical Coding for Extreme Low-Bit Large Language Model Compression 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前大语言模型（LLMs）在部署时面临严重的**内存容量、权重带宽和检查点存储压力**。低比特压缩是缓解这些问题的关键手段，但在极端低比特（如 ≤2 bits/weight）场景下，现有方法存在明显瓶颈：
- **标量/分组量化**（scalar/group-wise quantization）在极低比特下表示能力不足。
- **向量量化**（VQ）虽然表达能力强，但依赖显式的码本（codebook）、索引查找和额外存储开销。

因此，如何在**不引入显式码本的前提下实现高效且高保真的极低比特压缩**，成为核心挑战。

### 提出的新方法与创新思路
本文提出 **BISCo-LLM**（Lookup-Free Binary Spherical Coding for LLM），一种**无需码本的二值球面编码框架**，用于极端低比特 LLM 权重压缩。其核心创新包括：

#### （1）**无码本二值球面编码**（Codebook-Free Binary Spherical Coding）
- 将局部权重块映射到单位超球面上，并进行二值化为紧凑的球面码（spherical codes）。
- 存储的是**位打包的符号流**（bit-packed sign stream），而非显式的 VQ 聚类中心，避免了码本存储和索引查找。

#### （2）**残差 BSQ 编码阶段**（Residual BSQ Stage）
- 引入第二阶段 BSQ 编码器来建模第一阶段重建后的残差误差。
- 显式构建了一条**率失真路径**（rate-distortion path），在不增加码本的情况下提升重建精度。

#### （3）**类别级恢复蒸馏**（Category-Wise Recovery Distillation）
- 在每个 Transformer 模块类别（如 `q_proj`, `mlp.up_proj` 等）替换后，执行以该类别为中心的蒸馏训练。
- 减少局部权重重建与整体模型行为之间的**错配**（mismatch），使恢复目标更贴近真实任务表现。

#### （4）**敏感通道保护机制**（Sensitivity-Aware Protection）
- 对少量对激活敏感的通道保留一个小型 8-bit 辅助通路（protected-channel path），其余部分仍用 2-bit BSQ 编码。
- 该路径仅作为稳定机制，其存储成本被单独计入总预算。

---

### 相比现有方法的优势

| 维度 | 传统方法（如 GPTQ, AQLM, QuIP#） | BISCo-LLM |
|------|-------------------------------|----------|
| **码本依赖** | 需要显式码本或投影矩阵 | ✅ 完全无码本，仅存二值码流 |
| **存储效率** | 码本 + 索引带来额外开销 | ✅ 更精确的存储核算（含 decoder、LoRA、metadata） |
| **表达能力** | 标量量化受限于离散层级数 | ✅ 球面编码提供方向性丰富表示 |
| **可扩展性** | 多码本难以跨层共享 | ✅ 类别级共享神经解码器，参数可摊销 |
| **恢复机制** | 层级蒸馏易忽略模块交互 | ✅ 类别级蒸馏更好反映全局影响 |

> ✅ **优势总结**：BISCo-LLM 实现了**更高保真度的 2-bit 压缩**，同时通过**去除显式码本**提升了部署友好性和存储透明性。

---

## 2. 核心实验方法和设置

### 使用的数据集

#### **预训练模型**
- **Qwen3-8B**
- **LLaMA3-8B**

#### **校准与蒸馏数据**
采用混合语料库进行 recovery distillation：
- Alpaca, OpenOrca（指令跟随）
- FineWeb-Edu（教育文本）
- RACE, SciQ（阅读理解 & 科学问答）
- LongAlpaca, LongAlign（长上下文任务）

> 数据总量约数十亿 token，确保覆盖多样化推理模式。

#### **评估基准**
- **Perplexity**: WikiText-2, C4
- **Zero/Few-shot Accuracy**:
  - BoolQ, RTE (GLUE)
  - WinoGrande
  - ARC-Easy / ARC-Challenge
  - OpenBookQA
  - PIQA
  - MMLU

---

### 实验设置与评估指标

#### 压缩粒度
- 对所有主要线性模块分类别压缩：
  - Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - MLP: `gate_proj`, `up_proj`, `down_proj`

#### 存储预算计算（Real Storage Budget）
明确报告以下组成部分：
```math
S_{\text{tot}} = S_{\text{code}} + S_{\text{dec}} + S_{8bit} + S_{\text{LoRA}} + S_{\text{meta}}
```
其中：
- $S_{\text{code}}$: 所有 BSQ 二值码流（base + residual）
- $S_{\text{dec}}$: 类别级神经解码器参数
- $S_{8bit}$: 受保护通道的 8-bit 存储
- $S_{\text{LoRA}}$: 可选 LoRA 补偿适配器
- $S_{\text{meta}}$: 缩放因子、偏移、索引等元数据

> ⚠️ 这种细粒度核算使得与其他方法（如 AQLM、LiftQuant）公平比较成为可能。

#### 主要比特率
- **主压缩路径**：~2 bits/weight（BSQ + 残差）
- **辅助路径**：1% 通道使用 8-bit（$S_{8bit}$ 单独计费）

---

### 基线方法对比

分为两类：

#### （1）标量/分组量化方法（Scalar/Group-wise PTQ）
- GPTQ
- SpinQuant
- OSTQuant
- QuIP
- AWQ

#### （2）向量/结构化编码方法（Vector/Structured-Coding）
- AQLM（Additive Quantization）
- VPTQ（Vector PTQ with Hessian）
- QuIP#
- LiftQuant（维度提升 + 投影）
- UniSVQ（统一标量-向量量化）

> 所有对比均基于相同或相近的实际存储预算（real bpw）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Qwen3-8B）

| 指标 | FP16/BF16 | BISCo-LLM |
|------|-----------|------------|
| **WikiText-2 Perplexity** | 9.73 | **10.18** |
| **平均下游准确率**（7项任务） | 69.92 | **68.05** |
| **性能差距** | — | **↓1.87 pts** |

> ✅ 在 ~2 bits/weight 下保持接近全精度的表现。

---

### 与基线方法的对比结果（Qwen3-8B）

#### Perplexity（越低越好）
| Method | WikiText-2 PPL |
|--------|----------------|
| GPTQ | 4.68×10⁴ |
| SpinQuant | 17.82 |
| QuIP# | 12.37 |
| UniSVQ | 14.82 |
| **BISCo-LLM** | **10.18** ✅ |

> 📈 **显著优于所有标量与向量基线**，逼近理论极限。

#### 平均准确率（越高越好）
| Method | Avg. Score |
|--------|------------|
| GPTQ | 37.29 |
| SpinQuant | 52.61 |
| AQLM | 64.94 |
| QuIP# | 67.55 |
| UniSVQ | 67.94 |
| **BISCo-LLM** | **68.05** ✅

> 🏆 **达到最高平均准确率**，超越 UniSVQ (+0.11)，QuIP# (+0.51)，AQLM (+3.11)。

#### 特定任务领先优势
- **ARC-Challenge**: 51.88 → 领先 QuIP# (46.50) 和 AQLM (45.22)
- **ARC-Easy**: 78.87 → 显著高于其他方法

> 表明 BISCo-LLM 不仅降低重建误差，更能保留复杂推理能力。

---

### 消融实验结果（Ablation Studies）

#### （1）残差 BSQ 的有效性（Table I）
在 LLaMA-2-7B 上测试单阶段 vs 双阶段 BSQ：

| Category | ΔPPL (Two-stage − One-stage) |
|---------|------------------------------|
| `up_proj` | -0.21 |
| `down_proj` | -0.21 |
| `gate_proj` | -0.20 |
| 平均 | **-0.13** ✅ |

> ✅ 残差编码在绝大多数模块中带来增益，尤其对 MLP 更有效。

#### （2）保护通道的作用（Table II）
启用 1% 8-bit 保护通道后：
- **平均 PPL 从 5.95 → 5.82**
- 最大收益来自 `down_proj`（6.58 → 6.10）

> ✅ 极小代价即可显著提升稳定性。

#### （3）解码器设计的影响（Table IV）
| Decoder Type | Overall Avg. |
|-------------|---------------|
| Symmetric (default) | 62.82 |
| Linear (no residual block) | 58.74 ↓ |
| Linear ×3 stages | 58.42 ↓ |

> ❌ 即便增加残差阶段也无法弥补非线性表达能力损失 → **非线性对称解码器至关重要**。

#### （4）保护轴选择（Table V）
| 方式 | Average PPL |
|-----|--------------|
| 输入通道保护（input-channel） | **5.82** ✅ |
| 输出通道保护（output-channel） | 5.89 |

> ✅ 输入通道更适合注意力模块；MLP 使用耦合中间通道保护。

#### （5）保护比例分析（Table VI）
| Ratio | Avg. PPL | Gain from 0→1% |
|-------|----------|----------------|
| 0% | 6.32 | — |
| 1% | 5.98 | 81.7% of total gain |
| 3% | 5.91 | +0.07（边际递减） |

> ✅ **1% 已捕获绝大部分收益**，支持轻量设计原则。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **无码本球面编码可行且高效**  
   证明了在 LLM 权重压缩中，**无需显式码本也能实现高质量重建**，通过二值符号流 + 共享神经解码器即可完成。

2. ✅ **残差结构显著提升率失真性能**  
   单纯扩大一阶段码空间利用率极低（<10⁻¹³），而将额外比特分配给残差结构能更有效地利用容量。

3. ✅ **类别级蒸馏优于层级蒸馏**  
   图 3 显示：即使每层 loss 收敛，拼接后的完整模型性能仍严重下降。类别级蒸馏能更好地协调模块间交互。

4. ✅ **极小保护通道即可大幅提升鲁棒性**  
   仅 1% 的 8-bit 通道就能吸收大部分敏感误差，是一种高效的“安全阀”机制。

5. ✅ **真实存储核算至关重要**  
   名义比特率（nominal bpw）不能反映实际开销。必须将 decoder、LoRA、metadata 显式计入才能公平比较。

---

### 方法的局限性

1. 🔒 **优化强度较高**
   - 需要类别级 BSQ 编码器训练 + 多轮 recovery distillation。
   - 不适用于一次性、快速压缩场景。

2. ⚙️ **未充分利用二值结构进行加速**
   - 当前仍需 materialize 解码后的浮点权重。
   - 未实现直接从 binary code 进行 fused computation（如 bit-level matmul）。

3. 📏 **固定码率策略**
   - 所有模块统一使用 ~2-bit 主路径 + 1% 8-bit 辅助路径。
   - 缺乏动态比特分配机制（如根据 Hessian 敏感度调整）。

4. 🧪 **尚未支持权重-激活联合量化**
   - 当前仅为 weight-only 压缩框架。
   - 无法进一步压缩激活和 KV Cache。

---

### 未来工作方向

1. **混合速率分配机制**（Mixed-Rate Allocation）  
   基于残差能量、Hessian 敏感度或激活统计，自适应地为不同模块/层分配更多残差比特或保护比例。

2. **融合式二值核函数**（Fused Binary Kernels）  
   开发可直接操作 binary spherical codes 的线性算子，跳过 materialization 步骤，实现真正高效的 inference。

3. **端到端权重-激活联合量化**  
   将 BSQ 思想扩展至激活张量，结合 SmoothQuant 或 AWQ 思路，实现全模型极低比特化。

4. **在线自适应压缩**（Online Adaptive Compression）  
   在推理过程中动态识别敏感通道并临时启用高精度路径，实现“按需解压”。

5. **扩展至 MoE 架构**  
   探索如何对专家网络（experts）进行差异化压缩，兼顾稀疏性与精度。

---

> 💡 **总体评价**：BISCo-LLM 是一项在**极低比特 LLM 压缩领域具有范式意义的工作**。它成功将视觉领域的 lookup-free binary coding 思想迁移到模型压缩中，提出了一个**兼具高性能、高透明度和工程实用性的新框架**，为未来“存储感知”的模型压缩研究指明了新方向。

</details>

---

### 5. [SQuaD-SQL: Efficient Text-to-SQL with Small Language Models via LLM-Guided Knowledge Distillation](https://arxiv.org/abs/2607.08161)

**Authors**: Wangyu Wu, Xiaojian Lin, Rong Fu, Zaiyang Yu, Xuhang Chen, Wenjun Yu, Zhenhong Chen  
**Category**: cs.CL  
**Published**: 2026-07-10  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.08161v1  

#### Abstract
Text-to-SQL is a fundamental task in natural language processing that enables users to interact with structured databases using natural language. While large language models (LLMs) have demonstrated remarkable performance on this task, their substantial computational requirements hinder deployment i...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《SQuaD-SQL: Efficient Text-to-SQL with Small Language Models via LLM-Guided Knowledge Distillation》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前 **Text-to-SQL** 任务中，尽管 **Large Language Models (LLMs)** 表现出色，但其高推理延迟、昂贵部署成本（需多块高端GPU）以及训练资源消耗大，严重限制了在 **resource-constrained environments**（如边缘设备、中小企业系统）中的实际应用。

同时，直接使用 **Small Language Models (SLMs)** 在该任务上表现较差，难以捕捉复杂的结构化推理逻辑。

### ✅ 提出的新方法：SQuaD-SQL
作者提出 **SQuaD-SQL**（Small-Qualified and Distilled for SQL），一种基于 **LLM-Guided Knowledge Distillation** 的高效训练框架，使 SLMs 能够接近 LLMs 的性能，同时显著提升效率。

#### 核心三要素：
1. **LLM-based Synthetic Data Generation**  
   利用 LLM（如 GPT-4o）作为“教师”，通过精心设计的 prompt 模板生成高质量的自然语言问句与对应 SQL 查询对（即 synthetic data），为 SLM 提供结构化监督信号。

2. **Parameter-Efficient Fine-Tuning (PEFT)**  
   采用 **LoRA** 技术进行微调，仅更新低秩矩阵参数，大幅减少可训练参数量，实现 **单张消费级 GPU（如 RTX 4090）即可完成全模型训练**。

3. **Domain-Adaptive Fine-Tuning**  
   针对特定领域生成领域相关的 synthetic data，增强模型在目标场景下的泛化能力。

### ✅ 相比现有方法的优势
| 维度 | 传统 LLM 方法 | SQuaD-SQL |
|------|----------------|-----------|
| 推理速度 | 慢（高延迟） | 快（轻量模型） |
| 内存占用 | 高（数十GB显存） | 低（可在24GB显存运行） |
| 训练成本 | 昂贵（多卡+全参数微调） | 极低（单卡+LoRA） |
| 性能 | SOTA但不可持续 | 接近 LLM 水平 |
| 可部署性 | 差 | 强，适合落地 |

> 🔍 **核心思想突破**：复杂符号推理能力（如 SQL 生成）不一定依赖模型规模，而可通过 **guided instruction + 结构化知识蒸馏** 在小模型中有效习得。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **WikiSQL**：主流 Text-to-SQL benchmark，包含约 80K 条 `(NL question, SQL, table)` 三元组。
  - 训练集：56,355
  - 验证集：8,421
  - 测试集：15,878
  - 特点：单表查询为主，SQL 包含 `SELECT`, `WHERE`, `GROUP BY` 等操作。

> 注：未使用更复杂的跨域多表数据集（如 Spider），聚焦于效率与通用性的平衡。

### ⚙️ 实验设置
- **基础模型**：`Qwen-1.5B`（15亿参数）作为学生模型（SLM）
- **教师模型**：`GPT-4o`
- **合成数据生成**：
  - 使用 GPT-4o 生成约 **50,000 条高质量 synthetic NL-SQL 对**
  - 输入包含 schema 信息（列名、示例行）、prompt 模板
- **微调策略**：
  - 使用 **LoRA**，rank $ r=8 $，scaling factor $ \alpha=32 $
  - 仅注入到 attention 层的 `q-proj` 和 `v-proj`
  - 批大小 4，学习率 5e-5，训练 10 轮
  - 使用 FP16 混合精度加速
- **硬件平台**：单块 **NVIDIA RTX 4090（24GB VRAM）**

### 📊 评估指标
1. **Execution Accuracy**：预测 SQL 执行结果是否与真实 SQL 一致（主要指标）
2. **Logical Form Accuracy**：预测 SQL 是否在逻辑形式上等价于标准答案

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（WikiSQL 测试集）

| 方法 | Dev 执行准确率 | **Test 执行准确率** |
|------|----------------|--------------------|
| GPT-4o（强基线） | 92.2% | **93.5%** |
| SQLNet | 69.8% | 72.1% |
| RAT-SQL | 74.1% | 76.3% |
| Qwen-1.5B（零样本） | — | 35.6% |
| Phi-3-mini（零样本） | — | 42.5% |
| **SQuaD-SQL（本文）** | **86.5%** | **86.9%** |

> ✅ **结论**：SQuaD-SQL 以仅 1.5B 参数的模型，在测试集上达到 **86.9% 执行准确率**，**超过所有传统模型和多数结构化 QA 模型**，距离 GPT-4o 仅差 ~6.6%，但推理更快、内存更低。

### 🔁 与基线方法对比
- **远超传统神经模型**（如 SQLNet、RAT-SQL）
- **碾压原生 SLMs**（如 Qwen-1.5B 零样本仅 35.6% → 微调后达 86.9%）
- **逼近 LLM 性能**，差距小于 7%，但具备数量级的效率优势
- **优于近期结构化问答模型**（如 TrustUQA: 85.9%, M3: 80.3%）

### 🔍 消融实验结果（Ablation Study）

| 方法 | Prompt Eng. | Distillation | Data Filter | 准确率 (%) |
|------|-------------|--------------|-------------|------------|
| Zero-shot |            |              |             | 35.6       |
| Ablation 1 | √          |              |             | 45.6       |
| Ablation 2 |            | √            |             | 80.3       |
| Ablation 3 |            |              | √           | 83.5       |
| Ablation 4 | √          | √            |             | 83.2       |
| **Full (Ours)** | √      | √            | √           | **86.9**   |

> 💡 发现：
- **知识蒸馏（Distillation）是最大增益来源**（+44.7 pts）
- **数据过滤** 提升 3.2 pts，去除噪声至关重要
- **Prompt Engineering** 单独作用有限，但与其他组件协同效果明显
- 三者结合带来 **累计增益 >60 pts**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **小模型也能学会复杂推理**：通过 LLM 提供的结构化教学信号（synthetic data + 自我评估），SLMs 可内化 SQL 推理模式，无需依赖大规模参数。
2. **知识蒸馏 + PEFT 是高效路径**：LoRA + 合成数据可在单卡上完成高质量微调，极大降低部署门槛。
3. **质量控制至关重要**：LLM 生成的数据存在噪声，引入语法解析、执行验证和 LLM 自评打分机制可显著提升下游性能。
4. **效率与性能可兼得**：SQuaD-SQL 在保持 **86.9% 高准确率**的同时，实现了 **快速推理、低内存占用、低成本训练**，适用于现实场景。

### ⚠️ 方法的局限性
- 当前验证集中在 **单表查询任务（WikiSQL）**，尚未扩展至复杂多表、跨域场景（如 Spider、BIRD）。
- 依赖 LLM 生成数据的质量，若教师模型本身有偏见或错误，可能被传递给学生模型。
- 仍需要一定量的 schema 信息构建 prompt，自动化程度有待提高。
- LoRA 的 rank 选择对性能敏感（图4显示 r=8 最优，过高反而下降）。

### 🔮 未来工作方向
1. 将 SQuaD-SQL 框架扩展至 **Spider、BIRD 等复杂 Text-to-SQL 数据集**
2. 探索 **迭代式自我改进机制**（如 Iterative Distillation），让 SLM 反馈优化 prompt 或筛选规则
3. 引入 **multi-turn interaction** 或 **feedback loop** 进一步提升鲁棒性
4. 应用于垂直领域（如医疗、金融）的 domain-specific Text-to-SQL 系统构建
5. 探索完全无需人工干预的 **全自动 synthetic data pipeline**

---

## ✅ 总结一句话
> **SQuaD-SQL 成功证明：通过 LLM 指导的知识蒸馏 + 参数高效微调，小语言模型可以在资源受限环境下实现接近大模型的 Text-to-SQL 性能，为实际部署提供了高效可行的新范式。**

</details>

---

### 6. [Rethinking Small VLM Quantization: From Component-Wise Analysis to Hardware-Aware Edge Deployment](https://arxiv.org/abs/2607.08029)

**Authors**: Hyeju Shin, Chorwon Kim, Ryangsoo Kim, Hark Yoo, Jaein Kim  
**Category**: cs.LG  
**Published**: 2026-07-10  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.08029v1  

#### Abstract
The emergence of vision language models with fewer than 3 billion parameters has accelerated the implementation of on-device multimodal intelligence. However, a detailed understanding of component-wise quantization remains a bottleneck for optimal deployment. This paper presents a systematic evaluat...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Rethinking Small VLM Quantization: From Component-Wise Analysis to Hardware-Aware Edge Deployment

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前关于 **Vision Language Models (VLMs)** 的量化研究大多集中在端到端的整体模型评估上，且多在高性能服务器级 GPU 上进行。然而，在资源受限的边缘设备（如 NVIDIA Jetson Orin）部署小型 VLMs（sVLMs）时，面临以下挑战：
- 不同组件（vision encoder、projector、LLM backbone）对量化敏感度不同；
- 硬件平台差异导致延迟、能效表现不一致；
- 缺乏系统性的量化策略指导。

本文旨在解决：**如何在异构边缘 SoC 平台上实现高效、准确的小型 VLM 部署？**

---

### 🚀 提出的新方法与新思路

1. **提出组件级量化分析框架（Component-Wise Quantization Framework）**
   - 将 VLM 拆分为三个独立模块：`vision encoder`, `projector`, 和 `LLM backbone`；
   - 设计六种量化配置（cfg0–cfg5），分别控制各组件的精度（FP16 / INT8 / INT4），以隔离其影响。

2. **构建硬件感知的评估体系**
   - 在两个异构边缘平台（Jetson Orin NX 与 AGX）上进行全面测试；
   - 引入多维度评估指标，超越传统 accuracy-only 分析。

3. **提出五个可验证假设（Hypotheses-Driven Evaluation）**
   - 通过实验逐一验证关于量化敏感性、延迟异常、资源-效率权衡等关键问题。

---

### 🔍 相比现有方法的优势

| 方面 | 传统方法 | 本工作 |
|------|--------|-------|
| 分析粒度 | 整体模型量化 | 组件级拆解分析 |
| 实验环境 | Server GPUs | 边缘 SoC（Jetson Orin） |
| 评估维度 | 准确率为主 | Accuracy + VRAM + Latency + IPJ（Intelligence-per-Joule） |
| 可解释性 | 黑箱式结果 | 支持归因分析（如 dequantization overhead 来源） |

> ✅ **优势总结**：首次实现了跨平台、细粒度、硬件感知的小型 VLM 量化行为系统分析，为实际部署提供可操作的设计指南。

---

## 2. 核心实验方法和设置

### 📚 数据集
- 使用 **MME Benchmark** 进行性能评估。
  - 包含认知（Cognition）与感知（Perception）两类任务；
  - 总分 2800 分，结合 ACC 与 ACC+ 指标。

---

### ⚙️ 实验设置

#### 硬件平台
| 平台 | Jetson Orin NX | Jetson Orin AGX |
|------|----------------|----------------|
| GPU | 1024-core Ampere, 32 Tensor Cores | 2048-core Ampere, 64 Tensor Cores |
| 内存 | 16GB LPDDR5 (102.4 GB/s) | 64GB LPDDR5 (204.8 GB/s) |
| TDP | 25W | 50W |

> 注：Jetson Orin Nano 因 OOM 被排除。

#### 模型选择（共5个）
| 模型 | 参数量 | Vision Encoder | LLM Backbone |
|------|--------|----------------|---------------|
| Qwen3-VL-2B | ~2.44B | SigLIP-2 | Qwen3 (MoE) |
| DeepSeek-VL2-Tiny | ~3.37B | SigLIP-So400m | DeepSeek-MoE |
| PaliGemma2-3B | ~3.6B | SigLIP-So400m | Gemma2 (Dense) |
| LLaVA-OV-0.5B | ~1.03B | SigLIP-So400m | Qwen2 (Dense) |
| Kosmos-2.5 | ~1.37B | Pix2Struct | Decoder-only |

> 特点：涵盖 MoE vs. Dense 架构、SigLIP vs. 非 SigLIP 视觉编码器。

#### 量化配置（见 Table 2）
| Config | Vision | Projector | LLM | 描述 |
|--------|--------|-----------|-----|------|
| cfg0 (Base) | FP16 | FP16 | FP16 | 基线 |
| cfg1 | FP16 | FP16 | INT4 | 测试 LLM INT4 鲁棒性 |
| cfg2 | FP16 | INT8 | FP16 | 测试 projector 敏感性 |
| cfg3 | FP16 | INT8 | INT4 | cfg1 + cfg2 |
| cfg4 | INT8 | FP16 | FP16 | 测试 vision encoder 敏感性 |
| cfg5 | INT8 | FP16 | INT4 | cfg1 + cfg4 |

> 所有量化均基于 **BitsAndBytes** 框架实现。

---

### 📊 评估指标

| 指标 | 定义 | 单位 |
|------|------|------|
| **Accuracy** | MME 总得分（感知 + 认知） | 分数（最高 2800） |
| **VRAM Footprint** | 推理过程中的峰值内存占用 | GB |
| **Latency** | 分解为：<br>• Vision Encoding Time<br>• Projector Time<br>• TPOT（Time Per Output Token） | ms |
| **Energy Consumption** | 使用 `tegrastats` 测量整机功耗 × 时间 | Joules (J) |
| **IPJ (Intelligence-per-Joule)** | 归一化准确率 / 能耗 | Score/J |

> 算法伪代码见 Algorithm 1，支持精确瓶颈定位。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总（来自 Table 1 & Table 15）

| 模型 | cfg0 MME (NX) | cfg1 MME (NX) | VRAM ↓ (%) | TPOT ↑ (%) |
|------|----------------|----------------|------------|-------------|
| Qwen3-VL-2B | 2021.78 | 2077.82 (**+2.77%**) | 47.5% ↓ | 55.8% ↑ |
| DeepSeek-VL2 | 1911.12 | 1952.54 (+2.17%) | 48.5% ↓ | 71.1% ↑ |
| PaliGemma2 | 1661.01 | 1598.89 (**-3.74%**) | 46.5% ↓ | 10.4% ↑ |
| LLaVA-OV | 1361.83 | 1141.81 (-16.16%) | 24.3% ↓ | 49.1% ↑ |
| Kosmos-2.5 | 662.30 | 651.76 (-1.59%) | 13.5% ↓ | 1.7% ↑ |

> 💡 **观察**：LLM INT4 显著降低 VRAM，但多数情况下增加 TPOT；部分模型（如 Qwen3）反而提升 accuracy。

---

### 🔁 与基线方法对比结果

| 对比项 | 发现 |
|--------|------|
| **准确性排名稳定性** | 所有模型在 NX 与 AGX 上的 MME 排名完全一致：<br>`Qwen3 > DeepSeek > PaliGemma > LLaVA > Kosmos`<br>✅ 支持 H5：accuracy ranking 是 platform-invariant |
| **延迟 profile 差异性** | 同一模型在 NX 与 AGX 上的 latency 和 power 表现显著不同<br>❌ latency 不具备跨平台可迁移性 |
| **能量效率差异** | Qwen3-VL 在 AGX 上 IPJ 达 14.45，是 NX 上（5.87）的 **2.5×**，体现平台优化潜力 |

---

### 🔍 消融实验结果（Ablation Study）

#### ✅ RQ1: LLM INT4 是否随规模减小而更敏感？（H1）
- ❌ **原假设被推翻**：并非参数越小越脆弱。
- ✅ **新发现（H1rev）**：架构决定敏感性。
  - MoE 架构（Qwen3, DeepSeek）在 INT4 下表现更好（+56.04, +39.50 分）；
  - Dense 架构（PaliGemma, LLaVA）严重退化（-62.12, -220.02 分）；
  → **MoE 更抗量化噪声**。

#### ✅ RQ2: SigLIP 编码器是否存在延迟异常？（H2）
- ✅ **存在显著延迟异常**：
  - SigLIP-So400m 模型（PaliGemma, DeepSeek, LLaVA）在 INT8 下 vision latency 提升 **2.43–4.66×**；
  - 非 SigLIP（Kosmos-2.5）仅提升 1.15–1.20×；
- ❗ 该现象与 accuracy 无关（甚至 accuracy 微升）；
- 🔍 根源：**BitsAndBytes INT8 kernel 与 SigLIP 结构在 Ampere 架构上的执行碎片化**（详见 Table 17）。

#### ✅ RQ3: VRAM 节省是否带来推理加速？（H3）
- ❌ **没有**。尽管 VRAM 下降 21.8–47.5%，但 TPOT 上升 10.4–56.3%；
- 原因：**dequantization overhead** 抵消了计算加速；
- 能耗也上升（Qwen3-NX +54.7%）；
→ **INT4 是 VRAM-saving 技术，非 latency-reduction 技术**。

#### ✅ RQ4: 复合量化误差是否可加？（H4）
| 配置 | 是否可加 | 说明 |
|------|----------|------|
| cfg3 (Proj-INT8 + LLM-INT4) | ✅ 近似可加 | 误差叠加基本线性（残差 < ±4） |
| cfg5 (Vis-INT8 + LLM-INT4) | ❌ 非可加 | 架构依赖性强：<br>• PaliGemma2：额外损失 16.33 分（超加性）<br>• DeepSeek-VL2：增益 8.03 分（次加性） |

→ **modality alignment path 存在复杂交互效应**。

#### ✅ RQ5: 准确性排序是否跨平台稳定？（H5）
- ✅ **完全一致**：所有模型在 NX 与 AGX 上的 MME 排名不变；
- ❌ 能效（IPJ）、latency、power profile 则高度平台相关；
→ **accuracy 可复用，但能效需重新调优**。

---

## 4. 关键结论和发现

### ✅ 主要发现总结

| 发现编号 | 内容概要 |
|---------|----------|
| **F1** | **量化敏感性由架构主导而非规模**：<br>MoE 架构（如 Qwen3）在 INT4 下不仅稳定，还能提分；Dense 架构则严重退化。 |
| **F2** | **SigLIP 编码器存在硬件特定延迟陷阱**：<br>INT8 量化导致 Ampere 架构上 vision latency 暴增（最高 ×4.66），源于 BitsAndBytes kernel 执行路径碎片化。 |
| **F3** | **LLM INT4 降低 VRAM 但拖慢生成速度**：<br>由于 dequantization overhead，TPOT 显著上升，能耗也可能增加。 |
| **F4** | **复合量化误差总体近似可加，但在视觉-语言对齐路径中呈现非线性**：<br>特别是 Vis-INT8 + LLM-INT4 组合，受架构影响大。 |
| **F5** | **模型 accuracy ranking 具有跨平台一致性，但能效 profile 高度平台依赖**：<br>推荐使用 IPJ 指标进行边缘部署优化。 |

---

### ⚠️ 方法的局限性

1. **仅限于 BitsAndBytes 框架**
   - 未覆盖其他量化后端（如 GPTQ、AWQ、SmoothQuant）；
   - 虽做了 AWQ sanity check（Table 18），但未纳入主分析。

2. **缺少训练感知量化（QAT）**
   - 所有实验均为 Post-Training Quantization（PTQ）；
   - QAT 可能缓解部分敏感性问题。

3. **未探索自动精度分配**
   - 当前为手动配置，未来可引入 NAS 或 RL 实现动态 bit-width 分配。

4. **仅测试静态 batch size=1**
   - 实际应用中可能涉及 batching，影响 KV cache 与吞吐。

---

### 🔮 未来工作方向

1. **扩展至更多量化格式**
   - 探索 **FP8**, **W8A8**, **INT4+NF4 混合精度**；
   - 评估新型硬件原生支持（如 Hopper FP8 Tensor Core）。

2. **开发硬件感知的自动化工具链**
   - 构建 **Neural Architecture Search (NAS)** 或 **Precision Allocation Agent**，实现模态自适应量化。

3. **深入研究 MoE 为何更鲁棒**
   - 分析 expert routing 如何隔离量化噪声传播路径。

4. **推动标准量化 kernel 优化**
   - 针对 SigLIP 等流行 encoder，定制高效 INT8 kernel，避免当前执行碎片化问题。

5. **发布开源工具**
   - 已公开 [EdgeQuant-VLMEvalKit](https://anonymous.4open.science/r/EdgeQuant-VLMEvalKit/)，鼓励社区共建硬件感知评测生态。

---

> 📌 **最终启示**：  
> “**Don’t just compress the model — co-design the model, quantization, and hardware stack.**”  
> 边缘智能部署不能只靠模型压缩，必须走向 **HW-SW Co-Design** 新范式。

</details>

---

### 7. [Infinity-Parser2 Technical Report](https://arxiv.org/abs/2607.07836)

**Authors**: Zuming Huang, Jun Huang, Kexuan Ren, Baode Wang, Weizhen Li, Jianming Feng, Yu Wang, Yichen Yao, Shijun Lin, Yige Tang, Cheng Peng, Weidi Xu, Wei Chu, Yinghui Xu, Yuan Qi  
**Category**: cs.AI  
**Published**: 2026-07-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.07836v1  

#### Abstract
We present Infinity-Parser2, a large multimodal model that couples a controllable data-synthesis pipeline with multi-task reinforcement learning for end-to-end document parsing, addressing the persistent scarcity of faithfully annotated parsing corpora. Our contributions are threefold. First, we bui...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Infinity-Parser2 Technical Report 核心总结

## 1. 论文的主要贡献和创新点

### 解决的问题
文档解析（Document Parsing）作为多模态理解的关键任务，长期面临两大瓶颈：
1. **数据稀缺性**：高质量、大规模、精确标注的文档解析语料库严重不足，尤其在复杂布局、低资源语言、化学公式、图表等领域。
2. **优化碎片化**：现有方法通常将文本识别、布局分析、表格解析、公式识别等视为独立任务或串行阶段，缺乏统一的优化信号，导致错误累积和跨任务泛化能力弱。

### 提出的新方法与创新
为解决上述问题，本文提出 **Infinity-Parser2**，一个结合可控数据合成与多任务强化学习（Multi-task Reinforcement Learning）的大规模多模态模型，其核心创新如下：

- **构建可扩展的数据合成引擎**：
  - 设计了一个基于 **DOM（Document Object Model）** 的文档合成引擎，通过可控渲染框架与迭代精炼循环，生成布局保真且标注精确的合成文档。
  - 发布了开源数据集 **Infinity-Doc2-5M**，包含 **500万** 个中英文双语样本，覆盖学术论文、财报、报纸等多种文档类型，并提供元素边界框、标准内容形式（Markdown, HTML, LaTeX, SMILES, 结构化图表）和全页阅读顺序。

- **引入可验证的多任务奖励系统**：
  - 提出了一种 **联合强化学习（Joint Reinforcement Learning）** 框架，支持 **8个协同训练的目标**：
    - 文档解析（Document Parsing）
    - 布局分析（Layout Analysis）
    - 表格解析（Table Parsing）
    - 数学公式解析（Math Formula Parsing）
    - 图表解析（Chart Parsing）
    - 化学公式解析（Chemical Formula Parsing）
    - 文档视觉问答（Document VQA）
    - 通用多模态理解（General Multimodal Understanding）
  - 每个任务均采用其原生评估指标作为 **可验证奖励（Verifiable Reward）**，避免了学习型奖励模型的偏差。

- **发布两个部署优化的模型变体**：
  - **Infinity-Parser2-Flash**：针对低延迟推理优化，吞吐量相比前代提升 **3.68倍**。
  - **Infinity-Parser2-Pro**：面向高精度场景设计，在多个基准上达到 **SOTA** 性能。

### 相比现有方法的优势
- **端到端统一架构**：单一模型处理从感知到推理的完整解析栈，避免了传统流水线方法的错误传播。
- **强大的泛化能力**：得益于多任务RL和高质量合成数据，在图表、化学公式、文档VQA等专业领域表现优异。
- **开放性与可复现性**：公开了完整的模型、代码和数据集（Infinity-Doc2-5M），推动社区发展。

---

## 2. 核心实验方法和设置

### 使用的数据集
| 类别 | 数据集 | 描述 |
|------|-------|------|
| **文档结构任务** | `Infinity-Doc2-5M` | 自研500万样本双语数据集，含丰富标注 |
| | `olmOCR-Bench`, `ParseBench`, `OmniDocBench-v1.6` | 公开文档解析基准 |
| **元素级解析任务** | `PubTabNet`, `FinTabNet` | 表格识别 |
| | `im2latex`, `UniMER` | 数学公式识别 |
| | `ChartSFT`, `ChartQA` | 图表解析 |
| | `CoSyn-Chemical`, `ChemDraw-Bench` | 化学公式识别 |
| **推理与泛化任务** | `DocVQA`, `AI2D`, `MathVista`, `MMMU` | 多模态理解与推理 |

### 实验设置
- **基础模型**：基于 `Qwen3.5` 架构。
  - Infinity-Parser2-Flash：基于 `Qwen3.5-2B`
  - Infinity-Parser2-Pro：基于 `Qwen3.5-35B-A3B`
- **训练流程**：
  1. **监督微调（SFT）**：在 Infinity-Doc2-5M 上进行一轮训练。
  2. **多任务强化学习（RLVR）**：使用 **Group Relative Policy Optimization (GRPO)** 进行后训练，每个输入生成8个输出（rollouts），并根据任务特定奖励进行优化。
- **硬件**：64块 NVIDIA H100 GPU，使用 Megatron-LM 进行分布式训练。

### 评估指标
| 任务 | 主要指标 |
|------|---------|
| 文档解析 | `olmOCR-Bench` 整体得分, `ParseBench` 整体得分 |
| 布局分析 | `mIoU`（平均交并比） |
| 表格解析 | `TEDS`（Tree-Edit-Distance-based Similarity） |
| 数学公式解析 | `CDM`（Character Detection Matching） |
| 图表解析 | `RMS-F1`, `AP@strict/slight/high` |
| 化学公式解析 | `InChI` 准确率, `Tanimoto` 相似度 |
| 文档VQA | `ANLS`（Average Normalized Levenshtein Similarity） |

### 基线方法对比
- **流水线方法**：MinerU2.5, PaddleOCR-VL-1.5
- **端到端专家模型**：DeepSeek-OCR-2, dots.ocr, olmOCR-2
- **通用多模态模型**：Qwen3.5-2B, Qwen3.5-35B-A3B, Gemini-3-Pro, GPT-5.5

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | olmOCR-Bench | ParseBench | OmniDocBench-v1.6 |
|------|--------------|------------|-------------------|
| **Infinity-Parser2-Pro** | **87.6** | **74.3** | **93.95** |
| Infinity-Parser2-Flash | 86.0 | 72.2 | 91.98 |
| DeepSeek-OCR-2 | 76.3 | 41.2 | 90.17 |
| PaddleOCR-VL-1.5 | 78.5 | 66.0 | 94.87 |
| MinerU2.5 | 75.2 | 45.9 | 92.98 |

- **Infinity-Parser2-Pro** 在 `olmOCR-Bench` 和 `ParseBench` 上均达到 **SOTA**，分别超越最强基线 **3.7** 和 **4.2** 个百分点。
- **Infinity-Parser2-Flash** 推理速度达 **1,624 tokens/sec**，相比前代 `Infinity-Parser-7B`（441 tokens/sec）实现 **3.68倍** 吞吐量提升。

### 与其他方法的对比
- **优于所有基线**：在文档解析、布局分析、表格、数学公式等任务上全面超越流水线、端到端专家和通用VLM。
- **在图表与化学公式上的优势**：
  - **Chart-to-Table** (`ChartQA`)：Infinity-Parser2-Pro 达到 **86.5 RMS-F1**，接近专用模型 TinyChart-3B (93.8)。
  - **化学公式识别** (`CoSyn-Chemical`)：InChI 准确率达 **53.91**，是当前最强的开源通用模型。

### 消融实验结果
#### （1）数据迭代飞轮（Data Iteration Flywheel）效果
| 阶段 | olmOCR-Bench | DocLayNet (mIoU) |
|------|--------------|-----------------|
| Round 1 (仅公开数据) | 28.4 | 49.03 |
| Round 2 (+伪标签网页数据) | 83.3 | 62.48 |
| Round 3 (+合成数据) | 85.3 | 62.31 |

- **伪标签真实文档** 是性能跃升的主因，带来 **54.9** 分提升。
- 合成数据对长尾和稀有布局有补充作用。

#### （2）训练策略消融
| 模型 | 配置 | olmOCR-Bench | DocLayNet | Chart2Table |
|------|------|--------------|-----------|-------------|
| Infinity-Parser2-Flash | SFT (ViT冻结) | 84.3 | 60.81 | 79.96 |
| | SFT (ViT开放) | 85.5 | 63.29 | 80.30 |
| | 多任务RL | **86.0** | **64.97** | **80.49** |

- 开放 **ViT编码器** 微调显著提升布局与解析能力。
- **多任务RL** 进一步提升几乎所有任务性能，验证了其有效性。

---

## 4. 关键结论和发现

### 主要发现
1. **可控数据合成是突破数据瓶颈的有效途径**：通过DOM引擎生成的合成数据，能够有效覆盖真实世界中的长尾分布和复杂布局。
2. **多任务强化学习能统一感知、结构与推理**：使用原生指标作为奖励，驱动模型在多个相关任务上协同优化，显著提升了泛化能力和鲁棒性。
3. **端到端架构优于传统流水线**：尽管在某些单项任务上可能略逊于专用模型，但其在综合文档理解和下游应用中的稳定性与一致性更具优势。

### 方法的局限性
1. **语言覆盖有限**：训练数据以中英文为主，其他语言或混合语言文档的解析性能会下降。
2. **对极端视觉复杂性的挑战**：
   - 密集重叠的图表系列。
   - 任意角度旋转的表格。
   - 超细字体或严重模糊的扫描件。
3. **格式保留不足**：未能完全保留原文档中的粗体、斜体、删除线等内联格式。
4. **指令跟随能力有限**：难以可靠执行复杂的多步骤视觉指令。

### 未来工作方向
1. **扩展数据迭代飞轮**：通过注册新的渲染器，持续添加新模态（如3D图表、手写笔记），逐步覆盖更多长尾场景。
2. **从页面级解析迈向文档级理解**：结合结构提取与下游推理，使模型能作为智能代理的可靠感知层。
3. **增强跨文档理解能力**：支持对多份关联文档（如年报与附注）进行联合分析。
4. **扩大语言和布局覆盖范围**：纳入更多低资源语言和非西方排版模式（如阿拉伯语右向左书写）。

</details>

---

### 8. [FedOPAL: One-Shot Federated Learning via Analytic Visual Prompt Tuning](https://arxiv.org/abs/2607.08368)

**Authors**: Lingyu Qiu, Daniela Annunziata, Stefano Izzo, Fabio Giampaolo, Francesco Piccialli  
**Category**: cs.AI  
**Published**: 2026-07-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.08368v1  

#### Abstract
With the widespread deployment of basic models in edge intelligence, communication bandwidth has become a core bottleneck restricting the scalability of federated learning. Although one-shot federated learning alleviates this problem by minimizing communication rounds, existing iterative fine-tuning...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FedOPAL: One-Shot Federated Learning via Analytic Visual Prompt Tuning

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **One-Shot Federated Learning (OFL)** 场景中，通信效率是核心瓶颈。虽然已有方法如基于知识蒸馏（KD）或数据合成的方法减少了通信轮次，但通常需要服务器端进行迭代训练，导致**服务器计算开销大、超参数敏感**。

另一方面，**Analytic Federated Learning (AFL)** 通过闭式解（closed-form solution）实现零梯度、单轮聚合，极大提升了效率。然而，AFL 的理论有效性依赖于一个强假设：各客户端从冻结骨干网络提取的特征必须是高质量且线性可分的。在现实中的 **Non-IID 数据分布**下，这一假设常被破坏，导致特征流形错位（feature manifold misalignment），严重损害模型性能。

### 提出了什么新方法或新思路
本文提出 **FedOPAL**（Federated One-shot learning via Analytic Visual Prompt Tuning），将 **Visual Prompt Tuning (VPT)** 引入 Analytic FL 框架，作为解决 Non-IID 下特征对齐问题的新范式。

其核心思想是：
- 在客户端使用 **VPT** 对输入嵌入序列注入可学习的视觉提示（visual prompts），以“轻量级”方式局部修正特征分布；
- 将这些提示视为**特征整流器（feature rectifiers）**，主动将异构数据映射到线性可分的空间，从而满足 AFL 的理论前提；
- 服务端仅需聚合统计量 $ R_k $ 和 $ C_k $ 并求解闭式分类器 $ W^* $，同时对提示进行平均得到全局提示 $ P_{\text{global}} $；
- 整个过程**无需服务器端训练**，实现真正的高效协作。

### 相比现有方法的优势
| 维度 | FedOPAL | 传统 KD/Generator 方法 | 原始 AFL 类方法 |
|------|--------|------------------------|----------------|
| **通信轮次** | 单轮（One-shot） | 单轮 | 单轮 |
| **服务器计算成本** | 零训练（仅矩阵运算） | 高（需迭代优化生成器或蒸馏） | 极低（闭式解） |
| **对 Non-IID 的鲁棒性** | 强（通过 VPT 主动校正特征） | 中等（依赖生成质量） | 差（静态特征假设失效） |
| **系统总负载** | 最小化 | 转移至服务器 | 极低但性能受限 |

> ✅ **核心优势**：在保持 **zero server-side training cost** 的前提下，显著提升 AFL 在 Non-IID 场景下的准确率，达到与 SOTA 迭代方法相当甚至更优的性能。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验覆盖多个具有不同语义特性的基准数据集，验证方法的泛化能力：
- **CIFAR-10 / CIFAR-100**：细粒度自然图像分类任务
- **SVHN**（Street View House Numbers）：街景数字识别
- **DTD**（Describable Textures Dataset）：纹理描述分类

### 实验设置和评估指标
#### 数据划分
- 采用 **Dirichlet 分布**（$ \alpha \in \{0.01, 0.1, 0.5, 1.0, 10\} $）控制标签非独立同分布程度：
  - $ \alpha \to 0 $：极端 Non-IID
  - $ \alpha \to \infty $：近似 IID
- 客户端数量默认设为 $ C = 10 $

#### 模型架构
- **主干网络**：CLIP-ViT-B/16（冻结）
- **本地可训练部分**：Visual Prompt Tokens（M=10）
- **分类器**：顶部线性层（由闭式解求得）

#### 评估指标
- **Top-1 Accuracy (%)**
- 跨不同 $ \alpha $ 设置下的稳定性分析
- 超参数敏感性测试（proximal coefficient $ \mu $, regularization $ \lambda $, local epochs $ E $）

### 基线方法对比
分为两类进行比较：

#### （1）KD/Generator-based OFL 方法
- Co-Boosting
- DENSE

#### （2）Pre-training based Analytic OFL 方法
- **AFL**：原始解析联邦学习
- **FedPFT**：参数化特征迁移
- **FedCGS**：结合原型学习的统计方法

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table I & II）

| 方法 | CIFAR-10 ($\alpha=0.1$) | CIFAR-100 ($\alpha=0.1$) | SVHN ($\alpha=0.1$) | DTD ($\alpha=0.1$) |
|------|--------------------------|----------------------------|---------------------|--------------------|
| AFL | 82.50 | 58.56 | 53.97 | 60.80 |
| FedCGS | 86.11 | 64.58 | 57.45 | 2.13 |
| FedPFT | 78.38 | 59.75 | 34.02 | 62.18 |
| **Ours (FedOPAL)** | **93.72** | **75.51** | **47.05** | **64.79** |

> 🔍 在 CIFAR-100 上超越最强基线 FedCGS 达 **+10.93%**；在 DTD 上避免了 FedCGS 的崩溃现象（从 2.13% 提升至 64.79%）。

### 与基线方法的对比结果
- **相比 Generator-based 方法（如 DENSE, Co-Boosting）**：
  - 后者在高度 Non-IID 下无法收敛（例如 Co-Boosting 在 CIFAR-10 上仅 20.78%）；
  - FedOPAL 利用 CLIP 的强大先验，在无数据合成的情况下仍取得高精度。
  
- **相比 Analytic 方法（AFL/FedCGS）**：
  - AFL 性能在所有 Non-IID 设置下基本持平（~82.5%），说明其无法适应分布偏移；
  - FedOPAL 随 $ \alpha $ 变化波动极小（如 CIFAR-10 上最大差值 < 1.5%），表现出卓越的**分布不变性**；
  - 在 SVHN 上略低于 FedCGS（因 CLIP 预训练偏向自然图像而非手写数字），但仍优于其他所有方法。

### 消融实验结果（Ablation Study）

#### （1）Proximal Coefficient $ \mu $ 的影响
- 控制本地 prompt 更新与初始状态之间的正则强度；
- 实验显示存在 **倒U型关系**：
  - $ \mu = 0.01 $：约束太弱 → 局部过拟合；
  - $ \mu = 1.0 $：约束太强 → 缺乏适应能力；
  - **最优区间为 $ \mu \in [0.1, 1.0] $**，平衡局部适配与全局一致性。

#### （2）Regularization 参数 $ \lambda $ 的敏感性
- FedOPAL 对 $ \lambda $ 极不敏感（变化范围从 $ 10^{-6} $ 到 $ 1.0 $，性能波动 < 0.5%）；
- 表明 VPT 优化后的特征空间本身具有良好条件数（well-conditioned），无需额外正则即可稳定求逆。

#### （3）Local Epochs $ E $ 的影响
- 即使只训练 $ E=1 $ 轮，也能达到接近最优性能；
- 显示出 **快速收敛特性**，适合资源受限边缘设备。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **Visual Prompt 可作为有效的特征整流工具**：在 Analytic FL 中引入 VPT，能主动纠正 Non-IID 导致的特征流形错位，使闭式解重新有效。
2. ✅ **无需服务器训练即可实现高性能聚合**：通过 dual aggregation（统计量 + prompts），实现了真正意义上的 **zero server-side training cost**。
3. ✅ **性能媲美 SOTA 迭代方法**：尽管仅为 one-shot，FedOPAL 在多数数据集上达到甚至超过复杂迭代方法的精度水平。
4. ✅ **对超参数和数据异构性高度鲁棒**：无论 $ \alpha $ 如何变化或 $ \lambda/\mu $ 如何调整，性能均保持稳定。

### 方法的局限性
1. ❗ **依赖强大的预训练模型（如 CLIP）**：若骨干网络不具备良好的 zero-shot 泛化能力，prompt 的调节作用可能受限。
2. ❗ **在特定领域（如 SVHN）表现受限**：由于 CLIP 主要在自然图像上预训练，对手写数字等 out-of-domain 数据泛化较弱。
3. ❗ **仅适用于 vision-language 模型结构**：当前设计基于 ViT + prompt 插入机制，难以直接迁移到 CNN 架构。

### 未来工作方向
1. 🔄 探索 **dynamic prompt selection** 或 **sparse prompting** 以进一步降低通信开销；
2. 🔍 将该框架扩展至 **multi-modal federated learning** 场景（如图文联合推理）；
3. ⚙️ 结合 **lightweight fine-tuning** 策略（如 LoRA）与 prompt tuning，增强局部适应能力；
4. 🌐 研究在 **cross-device 与 cross-silo FL** 中的实际部署可行性，特别是在带宽极度受限的 IoT 场景。

---

> 💡 **总结一句话**：  
> **FedOPAL 成功弥合了高效 Analytic FL 与现实 Non-IID 环境之间的鸿沟，提出了一种“客户端微调提示 + 服务端解析聚合”的新工程范式，为大规模基础模型在边缘智能中的高效协同提供了可行路径。**

</details>

---

### 9. [The Illusion of Equivalency: Statistical Characterization of Quantization Effects in LLMs](https://arxiv.org/abs/2607.08734)

**Authors**: Baha Rababah, Cuneyt Gurcan Akcora, Carson K. Leung  
**Category**: cs.AI  
**Published**: 2026-07-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.08734v1  

#### Abstract
Post-training quantization is widely used to deploy large language models in resource-constrained settings, yet its evaluation relies almost exclusively on accuracy and perplexity. We show that these metrics fail to capture behavioral changes induced by quantization. We introduce correctness agreeme...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*The Illusion of Equivalency: Statistical Characterization of Quantization Effects in LLMs*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前对大语言模型（LLMs）**post-training quantization**（PTQ）的评估主要依赖于传统的性能指标如 **accuracy** 和 **perplexity**。然而，这些指标无法捕捉量化带来的**行为层面的变化**（behavioral changes），即模型在个体样本上的决策一致性是否被保留。

该论文指出，即使 accuracy 或 perplexity 表现稳定，量化后的模型可能已在决策层面显著偏离原始模型——这种现象被称为“等效性幻觉”（**illusion of equivalency**）。

### 提出了什么新方法或新思路
1. **Correctness Agreement**  
   - 一种新的**决策级评估指标**，用于衡量原始模型与量化模型在相同输入下**正确预测的重叠程度**。
   - 定义为：$ \text{CA}(c;\theta,D) = \frac{1}{M} \sum_{m=1}^M \mathbf{1}[z_m = 1 \land z_m^{(c)} = 1] $，其中 $ z_m $ 和 $ z_m^{(c)} $ 分别表示 base model 和 quantized model 在第 $ m $ 个样本上是否正确。
   - 优势：独立于绝对准确率，直接反映**行为一致性**。

2. **结构化扰动分析框架**
   - 将量化建模为作用于参数空间的操作符 $ T_c(\theta) $。
   - 引入统计量（mean, std, skewness, kurtosis）和分布差异度量（cosine similarity, KL divergence, KS statistic, Euclidean distance）来量化注意力权重的层间结构性漂移。

3. **细粒度敏感性分析**
   - 对 Attention 中的四个投影矩阵（Query, Key, Value, Output）进行分组件分析，揭示不同模块对量化的敏感性差异。

### 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|----------|-----------|
| 评估目标 | 聚合性能（accuracy/perplexity） | 决策一致性 + 结构稳定性 |
| 可解释性 | 黑箱式性能下降 | 明确识别敏感层与结构断裂点 |
| 实用价值 | 判断“能否用” | 判断“是否真的像原模型一样工作” |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **语言建模任务（Perplexity）**：
  - `WikiText-2`
  - `C4`
- **下游推理任务（Accuracy & Correctness Agreement）**：
  - `HellaSwag`：常识推理填空
  - `Winogrande`：代词消解与上下文理解
  - `ARC`（AI2 Reasoning Challenge）：科学类多步推理

### 实验设置
- **模型**：
  - Llama-3.2-3B
  - Vicuna-7B-v1.5
  - Mistral-7B-v0.1
  - Llama-3.1-8B
- **量化方案**（基于 `llama.cpp`）：
  - **Legacy Quantization**：Q8_0, Q5_0, Q4_0
  - **K-Quantization**：Q6_K, Q5_K, Q4_K, Q3_K, Q2_K
- **硬件平台**：8 × NVIDIA Tesla V100-SXM2 GPUs

### 评估指标
| 类型 | 指标 |
|------|------|
| 性能指标 | Perplexity ↓, Accuracy ↑ |
| 行为一致性 | **Correctness Agreement (CA)** ↑ |
| 权重分布变化 | Mean shift, Std drift, Skewness, Kurtosis |
| 结构偏差 | Cosine Similarity ↑, KL Divergence ↓, KS Statistic ↓, Euclidean Distance ↓ |

### 基线方法对比
- 所有量化版本均以对应原始 full-precision 模型为 base。
- 对比不同 bit-width 下的行为一致性与结构稳定性趋势。
- 无显式 retraining 或 QAT 设置，聚焦 pure PTQ 场景。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 4）
| Quant | Llama-3.2-3B CA (%) | Vicuna-7B CA (%) | Mistral-7B CA (%) | Llama-3.1-8B CA (%) |
|-------|---------------------|------------------|--------------------|----------------------|
| Base  | 55.5                | 58.9             | 61.6               | 60.9                 |
| Q8_0  | 41.4                | 46.7             | 48.2               | 45.6                 |
| Q6_K  | 41.1                | 46.1             | 48.1               | 45.7                 |
| Q5_K  | 41.0                | 46.3             | 47.9               | 45.9                 |
| Q4_K  | 40.9                | 46.3             | 47.7               | 45.8                 |
| Q3_K  | 39.9                | 46.9             | 47.7               | 46.0                 |
| Q2_K  | **38.5**            | **42.9**         | **47.3**           | **44.1**             |

> ✅ 观察：尽管 accuracy 下降有限（如 Llama-3.2-3B 从 55.5 → 48.7），但 **CA 显著更低且持续下降**，说明大量原本正确的预测不再一致。

### 与基线方法的对比结果
- **Perplexity 不可靠**：
  - 多个配置（如 Q3_K）在某些模型上甚至优于 base model（见 Table 2 & 3），但其 CA 最低。
  - 表明 **low perplexity ≠ behavioral fidelity**。
- **Accuracy 掩盖行为漂移**：
  - 如 Vicuna-7B 在 Q2_K 下 accuracy 仍达 53.7%，但 CA 仅为 42.9%，远低于 base 的 58.9%。
  - 说明模型“看起来还行”，实则已偏离原始决策路径。

### 消融实验结果
#### （1）统计特性漂移（Fig. 1–3）
- **高/中位宽（≥5-bit）**：统计特性基本保持（mean≈0, low skewness, stable kurtosis）
- **低位宽（≤4-bit）**：
  - Q3_K/Q2_K 导致：
    - skewness 波动剧烈
    - kurtosis 崩溃（heavy-tail loss）
    - mean 明显偏移
  - 小模型（Llama-3.2-3B）更敏感

#### （2）分布差异分析（Fig. 4–5）
- **Cosine similarity**：在 Q2_K 处急剧下降（尤其 Vicuna-7B）
- **KL divergence / KS statistic**：在 Q3_K 和 Q4_0 达到峰值，表明分布发生剧变
- **Euclidean distance**：变化较小，敏感性较低

#### （3）组件级敏感性分析
| 组件 | 敏感性等级 | 发现 |
|------|------------|------|
| **Query (Q)** | ⭐⭐⭐⭐☆ | 最易受扰动，skew/kurtosis 剧烈波动 |
| **Key (K)**   | ⭐⭐⭐⭐☆ | 同上，尤其在 Q3_K/Q2_K 下严重失真 |
| **Value (V)** | ⭐⭐☆☆☆ | 相对稳健，仅在 Q2_K 出现轻微退化 |
| **Output (O)**| ⭐⭐★☆☆ | 高 kurtosis 但整体稳定，压缩容忍度高 |

> 🔍 结论：**Q 和 K 投影是量化中最脆弱的部分**，应优先分配更高 bit-width。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **存在“等效性幻觉”**：
   - 即使 perplexity 和 accuracy 接近原始模型，**decision-level behavior 已显著漂移**。
   - 传统指标不足以保证行为一致性。

2. ✅ **Correctness Agreement 是必要补充指标**：
   - 揭示了 accuracy 无法捕获的模型不一致性。
   - 应作为 PTQ 评估的标准组成部分。

3. ✅ **非线性断裂点存在于低位宽区域**：
   - **Q4_K 是安全上限**，行为尚可接受；
   - **Q3_K 开始出现明显退化**；
   - **Q2_K 进入崩溃区（breakdown regime）**，结构与行为双重失真。

4. ✅ **Attention 组件敏感性异质性强**：
   - **Q/K 层高度敏感**，V/O 层相对鲁棒。
   - 支持未来采用 **mixed-precision quantization** 策略：对 Q/K 使用更高精度，对 V/O 更激进压缩。

5. ✅ **K-Quantization 比 Legacy 更精细但也更复杂**：
   - 中等 bit-width 下表现良好（Q5_K/Q6_K）
   - 极低位宽下（Q2_K）反而引入更大扰动

### 方法的局限性
- 当前分析集中于 **decoder-only Transformer 架构**，未涵盖 encoder 或 encoder-decoder 模型。
- **dequantization 后权重重建为 float**，未考虑实际部署中的整数运算误差累积。
- 正确性标签依赖于确定性 scoring rule，在开放生成任务中难以扩展。

### 未来工作方向
1. 🔄 将 **Correctness Agreement 扩展至生成任务**（如通过语义相似度判断输出一致性）
2. 🧠 设计 **structure-aware quantization schemes**，针对 Q/K 层动态提升精度
3. 📊 构建统一的 **LLM Quantization Benchmark**，整合 performance, efficiency, alignment, 和 behavioral consistency
4. 🔍 探索 **quantization-induced hallucination 或 safety behavior drift**

---

> 💡 **一句话总结**：  
> 本文揭示了仅靠 accuracy 和 perplexity 评估量化模型会陷入“等效性幻觉”，提出 **Correctness Agreement** 作为关键补充，并通过系统分析证明：**即使性能看似稳定，低位宽量化也会导致行为漂移，尤其是 Query 和 Key 投影最为敏感**。

</details>

---

### 10. [DominoTree: Conditional Tree-Structured Drafting with Domino for Speculative Decoding](https://arxiv.org/abs/2607.08642)

**Authors**: Saw S. Lin (Zhiqi Zhang), Jyh-Shing Roger Jang  
**Category**: cs.CL  
**Published**: 2026-07-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.08642v1  

#### Abstract
Speculative decoding accelerates LLM inference by drafting several tokens and verifying them in parallel. Block-diffusion drafters such as DFlash produce

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：DominoTree: Conditional Tree-Structured Drafting with Domino for Speculative Decoding**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现有的 **Speculative Decoding** 方法在推理效率上存在权衡：
- **Block-diffusion drafters**（如 DFlash）通过单次并行前向传播生成整个候选块，速度快但仅建模每个位置的**边缘分布（marginals）**，忽略了路径依赖性，导致 draft 质量受限。
- **Best-first tree 方法**（如 DDTree、CaDDTree）能构建 draft tree 并并行验证多个候选路径，提升接受长度 $ \bar{T} $，但其评分机制基于**因子化假设**（factorized formulation），无法利用 Domino 这类具有**路径依赖修正**（path-dependent correction）的 drafter 的条件信息。

**Domino** 引入了一个轻量级的 GRU-based 因果修正模块，在不增加主干计算成本的前提下提升了 draft 分布的条件准确性。然而，其官方实现仅用于生成单一链式 draft，**未发挥其路径依赖结构在树形 drafting 中的潜力**。

本文提出 **DominoTree**，首次将 Domino 的**非因子化、条件性 draft 分布**有效集成到 best-first draft tree 构建中，解决了“如何在保持高效的同时充分利用条件信息”的问题。

---

### **提出的新方法与创新思路**
#### **DominoTree：训练免费的条件性 draft tree 构建方法**
- **核心思想**：在 Domino 的 GRU 修正机制基础上，为每条从根到节点的路径重新运行 GRU 修正，得到该路径下的**条件 logits**，而非使用固定的边缘 logits。
- **技术实现**：
  - 复用 Domino 公开发布的 checkpoint 权重（GRU + correction head），无需额外训练。
  - 在 best-first heap 中，每个节点的子节点得分由其父路径上的 GRU 状态动态决定，打破了 DDTree 的因子化假设。
- **关键优化**：
  - **Candidate Restriction**：为降低每节点 GRU 修正的计算开销，限制修正范围为每个深度下边缘 top-$ M $ 的候选 token（$ M \ll |V| $），显著减少 FLOPs。
  - **GPU-native CUDA Graph Builder**：将每节点修正操作构建成 CUDA graph，消除 Python 驱动的 kernel launch 开销，使构建过程更高效且与 Python 实现**bit-identical**。

#### **CondAdaptive（负结果）**
尝试将 CaDDTree 的 cost-aware 自适应预算机制迁移到非因子化场景，但由于 GRU 修正后的路径概率被**过度高估**（over-credited），导致自适应规则几乎总是选择最大预算，未能带来增益。最终采用固定预算作为默认配置。

---

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **Draft 质量** | 利用路径依赖的条件分布，生成更符合目标模型预期的候选序列，显著提升平均接受长度 $ \bar{T} $。 |
| **效率** | 仅需重运行轻量级 GRU 和低秩 MLP 修正，共享主干表示，避免重复昂贵的 full-vocabulary 投影。 |
| **兼容性** | 完全基于公开 Domino checkpoint，无需再训练，即插即用。 |
| **工程实现** | GPU-native builder 显著降低 per-round 构建延迟，确保理论优势转化为实际吞吐提升。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
共 8 个基准数据集，覆盖三大任务类型：
- **Math**: GSM8K, MATH-500, AIME25
- **Code**: HumanEval, MBPP, LiveCodeBench
- **Chat/Instruction Following**: MT-Bench, Alpaca

---

### **实验设置与评估指标**

#### **模型**
- 主要测试模型：**Qwen3-4B** 和 **Qwen3-8B**
- Drafter：`Qwen3-4B-Domino-b16`（block size 16）
- Target：对应大小的 Qwen3 模型

#### **温度设置**
- $ T \in \{0.0, 0.5, 1.0\} $
- 对于 $ T > 0 $，**draft deterministic**（best-first），**target temperature-sampled**，保证 lossless。

#### **评估指标**
- **Speedup over AR decoding**：端到端加速比，定义为  
  $$
  \text{Speedup} = \frac{\bar{T} \cdot L_{\text{target}}}{T_{\text{draft}} + T_{\text{verify}}}
  $$
- **Mean Accept Length ($ \bar{T} $)**：每轮平均接受的 token 数。
- **Throughput Gain (%)**：与 baseline 相比的吞吐提升（paired-bootstrap 95% CI）。

#### **硬件环境**
- Qwen3-4B 实验：双 RTX 5080（16GB）
- Qwen3-8B 实验：单 RTX A6000（49GB）

---

### **基线方法对比**
| 方法 | 类型 | 特点 |
|------|------|------|
| **DFlash** | Block-diffusion | 单链 draft，仅使用边缘 logits |
| **DDTree(16)** | Factorized Tree | 基于 DFlash 边缘分布构建 best-first tree |
| **CaDDTree** | Cost-aware Tree | DDTree + 自适应预算机制 |
| **Domino (released)** | Chain-only | 使用 GRU 修正的单链 draft，官方最优配置（CUDA-graph） |
| **DominoTree(16)** | **Conditional Tree (Ours)** | 本文方法，node budget=16, $ M=64 $, GPU-native builder |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Qwen3-4B, T=0）**
| 方法 | Speedup | $ \bar{T} $ |
|------|--------|-------------|
| DFlash | 4.48× | 6.25 |
| DDTree(16) | 4.96× | 7.03 |
| CaDDTree | 4.97× | 7.03 |
| **Domino (released)** | **6.41×** | **7.17** |
| **DominoTree(16)** | **6.63×** | **7.98** ✅ |

- **最高接受长度**：DominoTree 在所有温度下均达到最高的 $ \bar{T} $（最高达 **10.7 tokens/round**）。
- **最高加速比**：在多个数据集上实现 **最高 speedup**，Overall 平均达 **4.81×**（T=0）。

---

### **与基线方法的对比结果**

#### **vs. Domino (released)**
- **吞吐提升**（paired-bootstrap 95% CI）：
  - Qwen3-4B: **+9.2% ~ +10.4%** across $ T \in \{0, 0.5, 1.0\} $
  - Qwen3-8B: **+3.7% ~ +5.9%**
- **尤其在 chat 类任务上优势明显**：Alpaca 上最高提升 **+22%**（T=0.5）

#### **vs. DDTree / CaDDTree**
- **Overall 吞吐全面领先**：
  - Qwen3-4B: **+7.67% (T=0), +5.18% (T=0.5), +2.55% (T=1.0)**，CI-clean
  - Qwen3-8B: **+24% (T=0)**，随温度升高逐渐持平或小幅落后（code 类任务除外）
- **唯一在所有温度下均优于 DDTree/CaDDTree 的方法**

#### **消融实验结果**

##### **(1) 条件评分 vs. 边缘评分（Cond@16 vs. Marg@16）**
- 控制 drafter、builder、verifier 不变，仅改变评分函数。
- 结果：**+9.2% 吞吐提升**（95% CI: [8.1%, 10.3%]），证明增益来自**路径条件性**，而非仅仅是“有 tree”。

##### **(2) GPU-native Builder 的影响**
| Builder | vs. DDTree (T=0) | vs. DDTree (T=1.0) | vs. Domino (T=0) |
|--------|------------------|--------------------|------------------|
| Python Reference | +1.99% | -2.69% ❌ | +3.83% |
| **GPU-native (default)** | **+7.67% ✅** | **+2.55% ✅** | **+9.21% ✅** |

- **Builder 是胜负手**：无 GPU-native 优化时，高温度下可能反超失败；启用后稳定胜出。

##### **(3) 候选宽度 $ M $ 与预算 $ n $**
- $ M=64 $ 已足够：进一步扩大至 full vocab 不提升 $ \bar{T} $
- 预算 $ n=16 $：适合 chat 场景；$ n=32 $ 在 math/code 更优，但 build 成本翻倍

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **条件性 draft tree 可行且高效**：利用 Domino 的部分条件结构（partial-conditional），可在不重跑主干的情况下构建高质量 conditional draft tree。
2. ✅ **路径依赖显著提升 draft 质量**：相比 marginal scoring，conditional scoring 带来 **+9.2% 吞吐提升**，是核心增益来源。
3. ✅ **工程优化至关重要**：**GPU-native CUDA graph builder** 是将理论优势转化为实际性能的关键，否则高 build 成本会抵消增益。
4. ✅ **超越所有现有方法**：在 Qwen3-4B/8B 上，DominoTree 在大多数设置下实现了**最高的 $ \bar{T} $ 和吞吐**，尤其是在 chat 和低温度场景。
5. ❌ **CondAdaptive 失败**：由于 GRU 修正后的路径概率被高估，自适应预算机制失效，最终采用固定预算。

---

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **依赖 Domino 架构** | 当前方法专为 Domino 设计，难以直接迁移到其他 drafter（如 EAGLE 或 JetSpec）。 |
| **未支持自适应预算** | 尽管尝试了 CondAdaptive，但因校准问题未能成功，仍使用固定预算。 |
| **非生产级系统** | 所有实验基于单流研究框架（HuggingFace Transformers），未集成到 vLLM/SGLang 等 serving 系统。 |
| **GPU 性能不可跨卡比较** | Qwen3-8B 实验在 A6000 上进行，与 5080 的绝对吞吐不具备可比性。 |

---

### **未来工作方向**
1. **开发校准良好的自适应预算机制**：针对非因子化 conditional tree，设计新的 budget selection 策略。
2. **多请求批处理与 serving 集成**：将 CUDA graph builder 扩展至 batched/multi-stream 场景，适配 vLLM/SGLang。
3. **探索更通用的 conditional drafting 框架**：不限于 Domino，支持更多类型的 causal correction 结构。
4. **结合 JetSpec 思路**：若未来出现训练免费的 causal parallel head，则可进一步统一 conditional drafting 范式。

---

> **一句话总结**：  
> **DominoTree** 首次将 Domino 的路径依赖修正机制引入 draft tree 构建，提出一种**训练免费、条件性、高性能**的 speculative decoding 方法，在保持工程可行性的前提下，实现了当前最优的 draft 质量与端到端吞吐。

</details>

---

### 11. [KronQ: LLM Quantization via Kronecker-Factored Hessian](https://arxiv.org/abs/2607.07964)

**Authors**: Donghyun Lee, Yuhang Li, Ruokai Yin, Priyadarshini Panda  
**Category**: cs.LG  
**Published**: 2026-07-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.07964v1  

#### Abstract
Post-training quantization (PTQ) is a widely adopted technique for compressing large language models (LLMs) without retraining. Existing second-order PTQ methods, including GPTQ, construct quantization objectives exclusively from input activation statistics, effectively assuming that all output chan...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：KronQ: LLM Quantization via Kronecker-Factored Hessian

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **Post-Training Quantization (PTQ)** 方法（如 GPTQ、GPTAQ）在量化大语言模型（LLMs）时，仅利用输入激活的协方差 $ H_x $ 作为 Hessian 的代理，**忽略了输出侧的梯度统计信息 $ H_G $**。这种做法隐含地假设所有输出通道对重建误差的贡献是均等的，而实际上不同输出通道的敏感度差异巨大（见图1(a)），导致在超低位宽（尤其是2-bit）下出现量化退化甚至发散。

此外，在 **mixed-precision 分配** 中，由于 Q/K/V 投影共享相同的输入，仅依赖 $ H_x $ 的方法无法区分它们之间的敏感度差异，造成次优的比特分配。

### 提出了什么新方法或新思路
本文提出 **KRONQ**，一种基于 **Kronecker-Factored Hessian (K-FAC)** 近似的新型 PTQ 框架，其核心创新如下：

1. **Kronecker-Factored Quantization Error**  
   在 K-FAC 近似下，完整的 Hessian 被分解为 $ H \approx H_x \otimes H_G $，其中 $ H_G = \mathbb{E}[gg^\top] $ 是输出梯度的协方差。KRONQ 将 $ H_G $ 引入量化目标函数中，从而联合考虑输入和输出两侧的二阶信息。

2. **Bidirectional Incoherence Processing (BiIP)**  
   扩展了传统的输入侧随机旋转（如 QuIP），引入**双向非相干处理**：
   - 输入侧：使用 $ H_x $ 进行对角重缩放和正交变换（Hadamard）
   - 输出侧：使用 $ H_G $ 进行对角重缩放和正交变换
   该方法显著降低了权重矩阵在输入和输出维度上的幅度方差（CV），提升了量化稳定性。

3. **Inter-Layer Mixed-Precision via Joint Hessian Traces**  
   提出新的子层敏感度评分：  
   $$
   s_l = \text{tr}(H_{G,l}) \cdot \text{tr}(H_{x,l})
   $$  
   该评分能有效打破 Q/K/V 投影间的敏感度退化（因为它们接收不同的下游梯度），实现更优的混合精度比特分配。

### 相比现有方法的优势
- **更高的量化精度**：尤其在 2-bit 和 3-bit 权重量化下表现远超 GPTQ/GPTAQ。
- **更强的鲁棒性**：在 LLaMA-3-70B 等存在极端 outlier 的模型上仍能稳定收敛，而 GPTQ/GPTAQ 发散。
- **无需修改求解器**：$ H_G $ 在列式 OBS 更新中代数抵消，因此可直接复用 GPTAQ 的高效求解器，保持计算效率。
- **免费的混合精度策略**：敏感度评分仅需一次反向传播预计算，无额外搜索开销。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **校准数据集（Calibration）**：`WikiText-2`，使用 128 个样本，上下文长度 2048。
- **评估数据集**：
  - **困惑度（Perplexity）**：`WikiText-2`
  - **零样本准确率（Zero-shot Accuracy）**：`PiQA`, `ArcC`, `ArcE`, `HellaSwag (HS)`, `WinoGrande (WG)`, `BoolQ`, `OpenBookQA (OBQA)`
  - **高难度推理基准**：`GPQA-Diamond`, `MMLU`, `AIME-2024`, `LiveCodeBench`

### 实验设置和评估指标
- **模型范围**：LLaMA-2 (7B/13B/70B)，LLaMA-3 (8B/70B)，以及 Gemma-3, DeepSeek-R1-Distill-Llama-8B, Phi-4-mini 等较新模型。
- **量化配置**：
  - **权重量化**：W4/W3/W2（每通道标量量化）
  - **分组量化（Group Quantization）**：g=128
  - **权值+激活量化（WxA4）**：结合 QuaRot 框架
  - **混合精度（Mixed-Precision）**：从 W2 升级部分子层至 W3
- **评估指标**：
  - 主要：`WikiText-2 PPL ↓`
  - 辅助：多个推理任务的 `zero-shot accuracy ↑`
  - 效率：`peak VRAM`, `decoding latency (TPOT)`

### 基线方法对比
- **主流 PTQ 方法**：GPTQ, GPTAQ, QuIP, QuIP#, BoA, SpinQuant, AWQ, OmniQuant
- **混合精度方法**：SliM-LLM, CMPQ, AMQ, Q-Palette, HIGGS
- **相关工作**：YAQA（同用 $ H_x \otimes H_G $，但求解器复杂）、GuidedQuant

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 权重量化（WxA16）—— LLaMA-3-70B 结果惊人
| 方法       | W4 PPL | W3 PPL     | W2 PPL     |
|------------|--------|------------|------------|
| GPTQ       | 2.85   | 27.49      | 2.6e3      |
| GPTAQ      | 2.85   | 399.46     | NaN (发散) |
| **KRONQ**  | 2.85   | **3.25**   | **7.93**   |

> 💡 在 LLaMA-3-70B 上，GPTQ/GPTAQ 在 W2 完全失效，而 KRONQ 以 **7.93 PPL** 实现可用量化。

#### ✅ 综合性能领先（LLaMA-2-7B, W2）
| 方法       | Wiki2 PPL | PiQA | ArcC | WG  | Avg Acc |
|------------|-----------|------|------|-----|---------|
| GPTQ       | 31.11     | 63.0 | 28.4 | 53.9| 48.4    |
| GPTAQ      | NaN       | NaN  | NaN  | NaN | NaN     |
| **KRONQ**  | **8.15**  | 68.8 | 29.1 | 61.0| **52.9**|

> 📈 KRONQ 不仅 PPL 更低，零样本准确率也全面领先。

#### ✅ 分组量化（Group Quantization, g=128）
在 LLaMA-2-7B W2 下：
- GPTQ: 274.0 PPL
- GPTAQ: 23.19 PPL
- **KRONQ: 7.61 PPL**

> 🔥 KRONQ 在分组量化下依然大幅领先。

#### ✅ 权值+激活量化（W2A4）
| 模型         | GPTQ  | GPTAQ | **KRONQ** |
|--------------|-------|-------|-----------|
| LLaMA-2-7B   | 36.74 | 10.91 | **9.38**  |
| LLaMA-2-13B  | 12.55 | 8.41  | **7.77**  |
| LLaMA-3-8B   | 32.79 | 19.14 | **16.47** |

> ⚡️ 显著降低激活量化带来的性能损失。

#### ✅ 新模型泛化能力
在 Gemma-3-12B-IT 上 W2：
- GPTQ: 7.1e3 PPL
- GPTAQ: 94.08 PPL
- **KRONQ: 54.39 PPL**

> ✔️ 泛化到 LLaMA 外的新架构仍具优势。

### 消融实验结果（Ablation Study）

| 变体                     | L2-7B PPL | L2-13B PPL | L3-8B PPL |
|--------------------------|-----------|------------|-----------|
| Base (GPTQ)              | 9.81      | 7.95       | 14.52     |
| Base (GPTAQ)             | 8.15      | 6.99       | 11.92     |
| w/o Scaling ($S_x, S_G$) | 8.44      | 7.10       | 12.94     |
| BiIP w/ $H_x$ only       | 8.47      | 7.13       | 12.56     |
| BiIP w/ $H_G$ only       | 180.43    | 81.18      | 14.55     |
| **Full KRONQ (BiIP)**     | **8.15**  | **6.99**   | **11.92** |

> 🔍 结论：
> - GPTAQ 基础优于 GPTQ
> - 对角重缩放（Scaling）是必要的
> - 仅输出侧旋转无效
> - **双向非相干处理（BiIP）效果最佳**

---

## 4. 关键结论和发现

### 主要发现
1. **输出侧梯度协方差 $ H_G $ 至关重要**：忽略 $ H_G $ 会导致在低位宽（尤其是2-bit）下严重性能下降甚至发散，特别是在存在极端 outlier 的模型（如 LLaMA-3-70B）上。
2. **双向非相干处理（BiIP）显著提升稳定性**：同时在输入和输出维度进行旋转和重缩放，能有效抑制权重和 Hessian 的相干性，降低量化误差。
3. **tr($H_G$)·tr($H_x$) 是有效的混合精度评分**：相比仅用 tr($H_x$)，该评分能更好地区分 Q/K/V 等共享输入的子层，实现更优的比特分配。
4. **KRONQ 在极低位宽下优势最大**：随着 bit-width 降低，传统方法性能急剧恶化，而 KRONQ 因利用了更完整的 Hessian 信息，表现出更强的鲁棒性。

### 方法的局限性
- **需要一次反向传播**：相比 GPTQ/GPTAQ，需额外进行一次 backward pass 预计算 $ H_G $，增加离线计算成本。
- **峰值内存开销更高**：$ H_G \in \mathbb{R}^{d_{\text{out}} \times d_{\text{out}}} $ 是稠密矩阵，在 BiIP 预处理阶段带来临时内存压力（但之后释放）。
- **推理时有轻微开销**：需在线还原 Hadamard 变换，引入 $ O(d_{\text{in}} \log d_{\text{in}} + d_{\text{out}} \log d_{\text{out}}) $ 开销，与 QuIP# 相当。

### 未来工作方向
- 探索更高效的 $ H_G $ 近似方法（如低秩分解）以减少内存占用。
- 将 KRONQ 思路扩展到其他模型结构（如 Vision Transformer、MoE）。
- 结合训练感知量化（QAT）进一步提升性能。
- 研究如何将 BiIP 应用于激活量化以进一步压缩。

---

> ✅ **总结一句话**：  
> **KRONQ 通过引入 Kronecker-Factored Hessian 中的梯度协方差 $ H_G $，实现了双向非相干处理和更优的混合精度分配，在不改变核心求解器的前提下，显著提升了 LLM 在超低位宽下的量化性能与鲁棒性，尤其在 2-bit 场景下实现了突破性进展。**

</details>

---

### 12. [DeepSearch-World: Self-Distillation for Deep Search Agents in a Verifiable Environment](https://arxiv.org/abs/2607.07820)

**Authors**: Xinyu Geng, Xuanhua He, Sixiang Chen, Yanjing Xiao, Fan Zhang, Shijue Huang, Haitao Mi, Zhenwen Liang, Tianqing Fang, Yi R. Fung  
**Category**: cs.CL  
**Published**: 2026-07-10  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.07820v1  

#### Abstract
Training tool-use agents to improve from their own experience remains challenging, as supervised fine-tuning relies on fixed teacher-distilled trajectories, while sparse-reward reinforcement learning provides weak supervision for long-horizon interactions. We present DeepSearch-Evolve, a self-distil...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*DeepSearch-World: Self-Distillation for Deep Search Agents in a Verifiable Environment*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前的 **tool-use agents**（工具使用智能体）在自我进化（self-evolving）方面面临三大瓶颈：
- **监督信号静态**：基于监督微调（SFT）的方法依赖于固定模型生成的轨迹，性能受限于骨干模型能力，容易饱和。
- **奖励稀疏**：强化学习（RL）方法依赖最终任务成功与否的稀疏奖励，无法提供对中间步骤（如查询构造、工具选择、证据提取）的有效指导。
- **监督不可靠**：现有的 on-policy self-distillation（OPSD）需要细粒度的教师策略，但在非确定性环境中，工具调用的反馈不稳定，导致监督噪声大。

这些问题使得长视野（long-horizon）的 web agents 难以从自身经验中稳定地持续提升。

### 提出了什么新方法或新思路
本文提出两大核心组件：
1. **DeepSearch-World**：一个**确定性且可验证的虚拟环境**，构建于离线 Wikipedia 语料之上，支持可复现的搜索与页面阅读操作。
   - 包含 **42万个多跳 QA 任务**，通过实体级随机游走（entity-level random walks）构建。
   - 支持**过程级验证**（progress verification）、**基于环境的反思**（grounded reflection）和**失败恢复机制**，为 agent 提供可靠的中间监督信号。

2. **DeepSearch-Evolve**：一种**自蒸馏（self-distillation）框架**，允许 agent 在 DeepSearch-World 中迭代地：
   - 生成轨迹（trajectory generation）
   - 过滤高质量轨迹（rejection sampling & quality filtering）
   - 将带结构的“支架轨迹”（scaffold trajectory）转换为标准 ReAct 格式
   - 微调学生模型（SFT update）

该框架实现了无需依赖更强模型蒸馏的**自主进化闭环**。

### 相比现有方法的优势
- **摆脱强模型依赖**：不依赖更强大的闭源模型（如 GPT-4）进行轨迹蒸馏，仅利用自身在可验证环境中的经验即可持续提升。
- **提供密集可靠的过程监督**：通过环境提供的实体级进度验证和规则化反思，解决了 RL 中奖励稀疏和 OPSD 中监督噪声的问题。
- **可扩展性强**：环境完全离线、确定、可复现，支持大规模并行轨迹生成，适合长期迭代训练。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **训练数据**：
  - 自建 **420K 多跳 QA 任务池**，基于 Wikipedia 超链接图的实体级随机游走生成。
  - 构建了约 **1000万条目的离线 Wikipedia 语料库**，用于模拟搜索与浏览。
- **验证集**：
  - **DeepSearch-Val**：377 个高质量、证据可检索的多跳 QA 实例，用于训练过程监控。
- **测试基准**（共7个）：
  - **BrowseComp**（英文网页浏览）
  - **BrowseComp-ZH**（中文网页浏览）
  - **HLE**（专家级学术推理）
  - **GAIA**（通用 AI 助手任务）
  - **xBench**（职业对齐的生产力评估）
  - **HotpotQA**（多跳问答）
  - **Search-QA**（综合搜索问答）

### 实验设置和评估指标
- **骨干模型**：Qwen3.5-9B-Instruct
- **训练流程**：
  - 进行 **11 轮自进化循环**（self-evolving rounds）
  - 每轮生成 10,000 条轨迹，保留至少 4,000 条正确轨迹后触发训练
  - 使用 **重要性采样**（importance sampling）混合不同轮次的数据（衰减因子 γ=0.5）
  - 最终在真实工具上应用 **GRPO** 微调以缩小离线-在线差距
- **评估指标**：各基准任务的准确率（Accuracy），以百分比表示。

### 基线方法对比
- **闭源模型**：
  - OpenAI Deep Research
  - OpenAI-o3
- **开源模型**（均基于 7B–9B 规模）：
  - R1-Searcher, Search-R1, ZeroSearch, ASearcher
  - DeepResearcher, PokeeResearch, WebSailor
  - WebExplorer, Marco-DR, MiroThinker-v1.0, DeepDive
- **特别对比**：Qwen3.5-9B-Instruct（未训练的骨干模型）

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | BrowseComp | GAIA | HotpotQA |
|------|------------|------|----------|
| **DeepSearch-World-9B** | **31.2%** | **61.5%** | **93.4%** |

### 与基线方法的对比结果
- 相比骨干模型 **Qwen3.5-9B-Instruct**，在所有基准上均有显著提升：
  - **+23.8%** on BrowseComp
  - **+22.9%** on BrowseComp-ZH
  - **+9.0%** on HLE
  - **+37.6%** on GAIA
  - **+29.0%** on xbench
  - **+48.1%** on HotpotQA
- 在开源模型中表现**最具竞争力**，接近甚至超越部分闭源系统（如 OpenAI Deep Research 在 GAIA 上为 67.4%，本模型达 61.5%）。
- 在 **HotpotQA** 上达到 **93.4%**，表明其具备极强的多跳推理与证据组合能力。

### 消融实验结果
#### 表格 2：轨迹过滤消融（SearchQA 上的性能）
| 配置 | SearchQA 准确率 |
|------|----------------|
| 无任何过滤 | 46.4% |
| 仅拒绝采样（RS） | 54.9% |
| 仅质量过滤（QF） | 48.1% |
| **RS + QF（完整方法）** | **58.2%** |

👉 **结论**：**答案正确性验证是关键**，但结合质量过滤可进一步提升性能。

#### 表格 3：轨迹转换策略消融（DeepSearch-Val）
| 配置 | 准确率 |
|------|--------|
| 仅 SFT（无转换） | 25.0% |
| 完整 pipeline（含状态内化与反思重写） | **31.9%** |
| 移除反思重写 | 16.7% |
| 移除状态内化 | 23.5% |

👉 **结论**：将“支架轨迹”中的**状态更新**和**环境反思**转化为 agent 自身的 `<think>` 推理过程至关重要，尤其是**反思重写**防止了答案泄露。

#### 其他分析
- **工具使用行为**：DeepSearch-World-9B 平均交互 **18.0 步**（vs. Qwen3.5-9B 的 4.7 步），访问页面 **5.4 次**（vs. 0.9 次），表现出更强的持久探索能力。
- **高级能力评分**（Advanced Capability Score）：DeepSearch-World-9B 达到 **70%**（vs. 骨干模型 19%），在规划、记忆、自纠错等方面显著增强。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **可验证环境是实现 agent 自我进化的关键基础设施**：DeepSearch-World 提供了确定性、可复现、可验证的交互环境，使得 agent 可以从自身经验中获得**可靠的过程级监督信号**。
2. **自蒸馏无需依赖更强模型**：通过在可验证环境中迭代生成、过滤、转换和微调，agent 可以仅凭自身经验实现持续提升，打破了对闭源强模型的依赖。
3. **过程监督优于结果监督**：相比稀疏的最终奖励，环境提供的**实体级进度验证**和**规则化反思**能有效引导 agent 学习规划、记忆、错误恢复等认知行为。
4. **大规模多样化任务池至关重要**：420K 的任务池避免了过拟合，支持更稳定的自进化过程。

### 方法的局限性
1. **知识覆盖有限**：当前环境基于 Wikipedia，领域和时效性受限，难以覆盖最新或专业领域知识。
2. **更新机制保守**：采用的是演进式 SFT（evolving SFT），而非 RL 或 OPSD，可能牺牲了一定的探索灵活性。
3. **语言限制**：训练仅使用英文轨迹，中文任务（如 BrowseComp-ZH）性能相对较低。

### 未来工作方向
- 扩展可验证环境至更广泛的知识源（如新闻、专业数据库、实时数据）。
- 探索结合 RL 或 OPSD 的混合更新机制，在保持稳定性的同时增强探索能力。
- 研究如何将高级认知能力（如规划、错误恢复）注入 RL 训练过程。
- 构建多语言、多模态的可验证 agent 训练环境。

--- 

> **总结**：本文提出了 **DeepSearch-World + DeepSearch-Evolve** 框架，首次展示了在**无外部强模型干预**的情况下，通过**可验证环境中的自蒸馏**，实现 deep search agents 的有效自我进化。其实验结果证明，**确定性环境 + 过程监督 + 自蒸馏** 是一条通往可扩展、可持续 agent 自我改进的可行路径。

</details>

---

### 13. [Cross-seed explainability using Procrustes-conditioned Joint End-to-end Top-K Sparse Autoencoders](https://arxiv.org/abs/2607.08499)

**Authors**: Bendeg\'uz V\'aradi, Zolt\'an Kmetty  
**Category**: cs.CL  
**Published**: 2026-07-10  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.08499v1  

#### Abstract
We present a Procrustes-conditioned Joint End-to-end Top-K Sparse Autoencoder (SAE) for extracting cross-seed universal features from independently trained BERT models. Cross-seed feature universality is a fundamental challenge in mechanistic interpretability: because dictionary learning is non-conv...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Cross-seed explainability using Procrustes-conditioned Joint End-to-end Top-K Sparse Autoencoders

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文致力于解决 **cross-seed feature universality**（跨种子特征通用性）这一在 **mechanistic interpretability**（机械可解释性）中的根本挑战。由于 **dictionary learning 是非凸优化问题**，即使使用相同架构和训练数据，不同随机初始化种子（seed）训练出的 BERT 模型也会学习到**几何上错位的 feature spaces**，导致看似相同的语义概念被编码为完全不同的隐空间维度。

这种“**feature misalignment**”使得跨模型比较特征变得不可靠，限制了对语言模型内部机制的泛化理解。

---

### 🚀 提出的新方法
作者提出了一种名为 **Procrustes-conditioned Joint End-to-end Top-K Sparse Autoencoder (SAE)** 的新架构，其核心创新在于：

1. **Procrustes 对齐预处理**  
   在联合训练前，先通过 **Orthogonal Procrustes Rotation** 将两个独立训练 BERT 模型的激活空间进行几何对齐，使它们共享统一的坐标系。

2. **联合端到端 Top-K SAE 架构**  
   使用单一共享的 **Top-K SAE** 同时处理两个对齐后的模型激活，结合：
   - **Top-K sparsity**：避免 L1 正则带来的 shrinkage bias；
   - **End-to-end 下游任务优化**：通过 MSE 和 KL 散度损失保证功能一致性；
   - **Cross-seed 稀疏激活损失 $ \mathcal{L}_{\text{cross}} $**：直接最小化两模型稀疏编码间的差异；
   - **Auxiliary dead-feature revival loss**：复活“死神经元”。

> 🔍 创新本质：不是像以往那样用正则项“惩罚不一致”，而是**从源头上对齐几何空间后再提取特征**，实现更本质的一致性。

---

### ⚖️ 相比现有方法的优势

| 方法 | 缺陷 | 本文改进 |
|------|------|---------|
| 独立训练 SAE + 后处理匹配（如 Hungarian 匹配） | 仅做索引对齐，无法纠正深层几何偏移 | 引入 Procrustes 显式旋转对齐 |
| Feature Aligned SAE (MFR) | 多个 SAE 并行训练，依赖 MFR 正则，仍可能错位 | 单一 Joint SAE + 几何对齐，结构更简洁高效 |
| Anthropic Joint SAE | 无显式空间对齐，依赖训练过程隐式对齐 | 加入 Procrustes 条件，显著提升对齐质量 |

✅ **优势总结**：
- 更高 cross-seed 特征相关性（Pearson r）
- 更多满足 “universal” 定义的特征（r ≥ 0.7）
- 计算轻量、无需复杂正则设计
- 可解释性强，支持定性验证

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
三个英文文本分类基准数据集，均具备良好可解释性以支持人工验证：

| 数据集 | 任务类型 | 类别数 |
|--------|--------|-------|
| **SST-2** | 情感分析 | 2（正面/负面） |
| **Stanford Politeness** | 礼貌程度识别 | 3 |
| **TweetEval (Emotion)** | 推特情绪分类 | 5 |

---

### ⚙️ 实验设置

- **模型基础**：`bert-base-uncased`，共训练 **10 个不同 seed 的 BERT 模型**，组成 **5 对 seed pairs**。
- **SAE 层位置**：BERT 第 10 层（共 12 层），late layer 更具语义抽象性但也更易分裂。
- **SAE 参数**：
  - Dictionary size: 6144
  - Top-K sparsity: k = 32
  - Shared encoder-decoder 结构
- **Procrustes 对齐样本量**：500 个数据点计算旋转矩阵 $ W_{\text{align}} $
- **训练目标函数**：
  $$
  \mathcal{L} = \mathcal{L}_{\text{KL}} + \lambda_{\text{DS}} \mathcal{L}_{\text{DS}} + \mathcal{L}_{\text{local}} + \lambda_{\text{cross}} \mathcal{L}_{\text{cross}} + \lambda_{\text{aux}} \mathcal{L}_{\text{aux}}
  $$

---

### 📊 评估指标

| 指标 | 描述 |
|------|------|
| **Top-10 r / Top-100 r** | 跨模型前10/100个最活跃特征的平均 Pearson 相关系数 |
| **Univ. %** | 满足 r ≥ 0.7 且在两个模型中均激活的特征占比（占双激活特征总数） |
| **Dead %** | 在任一模型中激活少于 10 次的字典元素比例 |
| **Acc. change** | SAE 注入后下游任务准确率变化（越小越好） |
| **Per-token activation** | 逐 token 计算相关性，比句级平均更严格保守 |

---

### 🔁 基线方法对比

共六种条件进行消融研究：

1. **Independent SAEs (index)**：独立训练，按原始索引匹配（近零相关）
2. **Independent SAEs (matched)**：匈牙利算法匹配最优对应
3. **Independent SAEs + Procrustes (matched)**：后处理 Procrustes 对齐 + 匹配
4. **Joint – no rotation, no cross-loss**：联合训练但无对齐无交叉损失
5. **Joint – rotation only**：加入 Procrustes 对齐但关闭 $ \mathcal{L}_{\text{cross}} $
6. **Joint – full***（本文完整方法）：全组件启用

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1）

| Dataset | 方法 | Top-10 r | Top-100 r | Univ. % |
|--------|------|----------|-----------|---------|
| **Politeness** | Independent (matched) | 0.525 | 0.482 | 16.1% |
| | Joint – rotation only | 0.738 | 0.696 | 50.2% |
| | **Joint – full*** | **0.764** | **0.722** | **57.9%** ✅ |
| **SST-2** | Independent (matched) | 0.537 | 0.363 | 5.0% |
| | Joint – rotation only | 0.609 | 0.432 | 12.7% |
| | **Joint – full*** | **0.702** | **0.503** | **19.0%** ✅ |
| **TweetEval** | Independent (matched) | 0.646 | 0.511 | 9.8% |
| | Joint – rotation only | 0.859 | 0.828 | 57.6% |
| | **Joint – full*** | **0.865** | **0.846** | **58.3%** ✅ |

> ✅ 所有三项指标上，“Joint – full” 均达到最高值。

---

### 🔍 消融实验结果

#### （1）Procrustes 对齐是关键
- 单独用于独立 SAE（post-hoc alignment）效果有限甚至下降；
- 但在 **joint training 中作为前置条件时大幅提升性能**；
- 表明：**Procrustes 必须“condition”联合训练才有效**，而非简单后处理。

#### （2）$ \mathcal{L}_{\text{cross}} $ 提供小幅增益
- 在已对齐基础上进一步提升相关性（尤其 Top-100 r）；
- 但增益小于 Procrustes 本身，说明主要驱动力仍是几何对齐。

#### （3）Dead Neuron 现象加剧
- Joint 训练下 dead feature 比例上升（如 Politeness 从 56.5% → 71.1%），符合 Top-K SAE 文献观察；
- 使用 Auxiliary Loss 可部分缓解，尤其在长序列复杂的 Politeness 数据集中效果明显。

#### （4）Accuracy 影响极小
- 所有条件下的 Acc. change 均接近 0%，表明 SAE 注入未破坏原始模型性能。

---

### 🧪 验证实验补充（Appendix）

- **Permutation Test**：使用 30 组随机正交矩阵作为负样本，Procrustes 方案在所有实验中均优于随机旋转（empirical p < 0.032）；
- **Mismatched Control**：打乱 token 对应关系生成错误旋转矩阵，结果 Universality 几乎归零，证明对齐依赖真实结构；
- **Alignment Quality vs. Universality**：NRE（归一化残差误差）与 universal feature 数量强相关（R² = 0.704），说明对齐质量决定特征通用性。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Procrustes + Joint SAE 是实现 cross-seed universal features 的有效路径**  
   几何对齐 + 联合训练能系统性地提取出高相关性（r > 0.7）的通用特征。

2. **Post-hoc alignment 单独无效，必须与 joint training 结合**  
   证明了“**conditioning**”的重要性：空间对齐应作为训练的前提，而非事后补救。

3. **Top-K + End-to-End 设计提升了功能一致性**  
   避免 L1 偏差的同时，通过下游任务监督增强语义稳定性。

4. **Qualitative 分析证实可解释性**  
   如 Feature 389 被识别为“句首情态动词疑问句”（Hedged Indirect Request），并通过 counterfactual editing 得到验证。

---

### ⚠️ 局限性

1. **仅测试于 BERT 家族模型**  
   是否适用于其他 LLM（如 RoBERTa、LLaMA）尚待验证。

2. **局限于单一层级（layer 10）**  
   不同层级的 feature universality 可能差异大，早期层或有更高相关性但语义更浅。

3. **缺乏与其他先进方法直接对比**  
   如未与 **Feature Aligned SAE** 或 **Orthogonal SAE** 进行比较。

4. **Dead feature 问题依然存在**  
   尽管使用 AuxK，高 sparsity 导致大量字典元素无法激活。

---

### 🔮 未来工作方向

1. **扩展至更多模型架构与种子数量**，验证方法鲁棒性；
2. **探索多层联合对齐机制**，构建跨层通用 feature 字典；
3. **结合 causal tracing 或 intervention 方法**，进一步验证 feature 功能；
4. **将本方法应用于社会科学研究**，如跨文化语言模式挖掘；
5. **开发自动化 feature labeling pipeline**，提升大规模 interpretability 实用性。

---

## 总结

📌 本文提出了一个**轻量、高效、可复现**的方法来解决 **cross-seed feature misalignment** 问题。通过引入 **Procrustes-conditioned Joint Top-K SAE**，实现了目前最高的跨种子特征相关性和通用性，在多个 benchmark 上显著优于现有基线。该方法不仅推动了 mechanistic interpretability 的标准化进程，也为构建“通用概念字典”提供了可行路径。

</details>

---

### 14. [Evaluating the Effect of Frame Rate in Sequence-Based Classification of Autism-Related Self-Stimulatory Hand Idiosyncrasies](https://arxiv.org/abs/2607.07957)

**Authors**: Raunak Mondal, Peter Washington  
**Category**: cs.AI  
**Published**: 2026-07-10  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.07957v1  

#### Abstract
Autism spectrum disorder (ASD) affects over 75 million individuals worldwide, yet scalable computational methods for remote behavioral screening remain limited. This study addresses two complementary challenges in automated detection of autism-related self-stimulatory behaviors from video: (1) ident...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Evaluating the Effect of Frame Rate in Sequence-Based Classification of Autism-Related Self-Stimulatory Hand Idiosyncrasies*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本研究针对**自闭症相关自我刺激行为（self-stimulatory behaviors, "stimming"）的视频自动识别**中存在的两个核心挑战：
1. **现有方法性能有限**：以往多采用 CNN 模型处理静态帧，忽略了行为的时间动态特性，导致准确率不高（62–76%）。
2. **标注数据稀缺**：如 SSBD 数据集仅有 75 个视频，限制了深度学习模型的训练与泛化能力。

### 提出的新方法与新思路
1. **系统性评估序列模型在自闭症行为分类中的表现**：
   - 首次全面比较 **LSTM** 和 **GRU** 在手部 flapping 行为检测任务上的性能，并探索不同**帧采样间隔（frame sampling interval）** 对结果的影响。
2. **提出并量化多种数据增强策略的有效性**：
   - 在基于 **I3D transfer learning** 的框架下测试了 10 种数据增强技术，并通过**消融实验（ablation study）** 分析各方法对整体性能的边际贡献。
3. **引入个性化机器学习协议（personalized machine learning）**：
   - 对每个被试者单独建模，在单个视频内进行时序划分训练/测试，验证个体内部行为一致性是否支持模型迁移。

### 相比现有方法的优势
- **超越 CNN 基线**：LSTM 和 GRU 显著优于此前基于 CNN 的方法（最高达 98.75%，远超 76% 的 CNN 基线）。
- **优化计算效率**：发现最佳采样间隔为每 15 帧一次，可在保持高精度的同时减少约 93% 的计算开销。
- **提供实用指导**：明确指出 **upsampling** 是最关键的数据增强手段，为小样本临床视频分析提供了可复用的最佳实践路径。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Self-Stimulatory Behavior Diagnosis (SSBD) dataset** [4]
  - 包含 75 个儿童展示自闭症典型自我刺激行为的视频（平均时长约 90 秒）
  - 行为类别：arm flapping、head banging、spinning
  - 注释格式：XML 文件标注行为起止时间
  - 视频来源：公开网络资源（家长/护理人员同意发布），无需 IRB 审批

#### 实验一（架构与采样率比较）
- 子集：50 个包含 hand flapping 的视频
- 输入特征：从每帧提取的 **pose-derived geometric features**
- 采样间隔：1, 5, 15, 30, 45, 90 帧
- 输出序列长度：⌊90/k⌋

#### 实验二（数据增强与迁移学习）
- 使用完整 SSBD 数据集（75 videos, 3 classes）
- 双流输入表示：
  - **RGB arrays**（64 帧片段）
  - **Optical flow vectors**（TV-L1 算法计算）

---

### 实验设置与评估指标

| 组件 | 描述 |
|------|------|
| **模型架构** | - LSTM：一层 LSTM + Dropout(p=0.5) + Dense(Softmax)<br>- GRU：同上结构，仅替换为 GRU 层 |
| **训练配置** | TensorFlow/Keras；Adam 优化器；categorical cross-entropy 损失函数 |
| **评估协议** | - 实验一：80-20 train-test split<br>- 实验二 ablation：leave-one-out 方式排除单一增强方法<br>- 个性化 ML：每视频分 5 段，前 4 段训练，第 5 段测试（80-20 时序分割） |
| **评估指标** | Accuracy, Precision, Recall, Loss |

---

### 基线方法对比
| 基线方法 | 来源 | 性能（Accuracy） |
|--------|------|------------------|
| CNN-based models | Khodatars et al. [5] | 62–76% |
| Bag-of-Words + SVM | Rajagopalan et al. [4] | 47.1% |
| MobileNetV2 + F1-score | Lakkapragada et al. [25] | F1 = 84 |

> 本文提出的 LSTM/GRU 方法显著优于上述所有基线。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（实验一：序列模型 + 不同采样率）

| Model | Frame Interval | **Accuracy** | Precision | Recall | Loss |
|-------|----------------|-------------|-----------|--------|------|
| LSTM | 15 | **97.5%** | 0.975 | 0.975 | 0.1401 |
| GRU  | 15 | **98.75%** | 0.8108 | 0.750 | 0.5073 |

- **峰值性能出现在每 15 帧采样一次时**，低于或高于此值均导致性能下降。
- 过密采样（如每帧）可能引入噪声或冗余信息，造成轻微过拟合。
- 过稀疏采样（如每 90 帧）丢失关键时间动态，准确率降至 ~81%。
- GRU 虽参数更少、训练更快，但仍达到最高准确率（98.75%），具备部署优势。

> ✅ **结论**：**15 帧采样间隔是性能与效率的最佳平衡点**

---

### 数据增强实验结果（实验二）

#### （1）独立增强效果（Standalone Performance）

| Augmentation Method | **Accuracy (%)** |
|---------------------|------------------|
| **Horizontal Flip** | **48.78** |
| Downsample | 43.90 |
| Salt Noise | 39.02 |
| Vertical Flip | 34.15 |
| Elastic Transformation | 34.15 |
| Temporal Fit (200) | 29.27 |
| Upsample | 29.27 |
| Inverse Order | 26.83 |
| Pepper Noise | 24.39 |
| Temporal Elastic Trans. | 24.39 |

> 📌 **Spatial augmentations > Temporal augmentations**  
> 水平翻转最有效，因其符合“左右手 flapping 外观相似”的现实不变性。

---

#### （2）消融实验（Leave-One-Out Ablation Study）

| Excluded Method | Training Loss | Val Loss | Precision | Recall |
|------------------|---------------|----------|----------|--------|
| **Upsample** | **5.0101** ⬆️ | 1.9899 | 0.3000 | 0.2927 |
| Horizontal Flip | 3.6999 | 4.8482 | 0.2308 | 0.2195 |
| Downsample | 1.9608 | 3.1856 | 0.4865 | 0.4390 |

> 🔍 **关键发现**：移除 **upsampling** 导致损失最大上升 → **它是整个 pipeline 中最关键的增强组件**

---

### 个性化机器学习结果
- 协议：每人一个模型，视频内 80-20 时序分割
- 结果：**Mean Loss = 1.84**, **SD = 0.79**
- 低标准差表明个体行为模式具有较强时序一致性，支持 within-video 泛化。

---

## 4. 关键结论和发现

### 主要发现
1. **序列模型显著优于 CNN**：
   - LSTM 和 GRU 利用时间建模能力捕捉重复性动作模式，在 hand flapping 检测中分别达到 **97.5%** 和 **98.75%** 准确率，大幅超越 CNN 基线（≤76%）。
2. **最优帧采样间隔为 15 帧**：
   - 平衡了时间分辨率与计算成本，相比逐帧处理节省约 93% 计算量。
3. **GRU 是高效部署的理想选择**：
   - 参数更少、训练更快，且取得最高准确率（98.75%），适合边缘设备或移动健康应用。
4. **数据增强中 upsampling 最关键**：
   - 尽管其单独使用效果一般（29.27%），但在完整 pipeline 中不可或缺，移除后性能崩溃最严重。
5. **空间增强优于时间增强**：
   - 如 horizontal flip 符合实际场景的空间对称性，而 temporal elastic 或 inverse order 可能破坏周期性信号。
6. **个性化建模可行**：
   - 个体内部行为具有一致性，可用于短时校准后持续监测，缓解小样本群体建模难题。

---

### 方法的局限性
1. **数据集规模极小**：
   - SSBD 仅含 75 个视频，难以支撑稳健的交叉验证或统计显著性检验。
2. **缺乏交叉验证**：
   - 实验一未使用 k-fold CV，影响结果可靠性。
3. **视频质量异质性强**：
   - 来源于互联网公开视频，存在视角、光照、背景等不一致问题。
4. **部分视频不可访问**：
   - 因预处理错误导致有效样本进一步减少。
5. **迁移学习依赖通用 Kinetics 预训练**：
   - I3D 在 Kinetics-400 上预训练，非医学/行为领域专用，可能存在领域偏移。
6. **模型结构较简单**：
   - 未尝试 TCN、Transformer 等更新的时序建模架构。

---

### 未来工作方向
1. **开展 k-fold cross-validation 与统计检验**，提升结果可信度。
2. **构建更大规模、标准化采集的临床行为视频数据库**。
3. **探索 domain-specific 预训练模型**，例如在大规模儿童行为数据上预训练 I3D。
4. **比较 TCN、Transformer 等新型时序模型** 在该任务上的表现。
5. **扩展至多模态融合**（如结合音频、生理传感器信号）。
6. **实证研究个性化模型所需的最小 calibration duration**，推动真实世界落地。

--- 

> 💡 **总体评价**：  
> 本文为**数据稀缺场景下的临床行为视频分析**提供了坚实的实证基础和实用指南，特别是在**模型选择、采样策略、数据增强设计**方面给出了明确建议，具有重要的工程指导意义。

</details>

---

### 15. [Compete Then Collaborate: Frontier AI Teachers Build a Verifiable Curriculum to Improve a Coding Student Beyond Imitation](https://arxiv.org/abs/2607.08255)

**Authors**: Miseong Shawn Kim  
**Category**: cs.AI  
**Published**: 2026-07-10  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.08255v1  

#### Abstract
Large language models increasingly serve as teachers generating training data for smaller students. Prior multi-teacher knowledge distillation methods merge outputs without determining which frontier model teaches best, often relying on an LLM judge biased toward its own outputs. We introduce a comp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Compete Then Collaborate: Frontier AI Teachers Build a Verifiable Curriculum to Improve a Coding Student Beyond Imitation*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前 Large Language Model (LLM) 作为“教师”生成训练数据用于蒸馏到小型“学生”模型已成为标准做法，但仍存在两个未充分探索的问题：
1. **如何客观衡量哪个前沿模型是更好的教师**？传统依赖 LLM-as-judge 存在自偏好偏差（self-preference bias），即 GPT 更倾向于选择 GPT 的输出。
2. **多教师协作的最佳方式是什么**？是简单合并输出进行模仿学习（SFT），还是有更优机制？

### 🚀 提出的新方法与思路
作者提出 **"Competition-Then-Collaboration"（先竞争后合作）框架**，其核心流程如下：

- **Competition Phase（竞争阶段）**  
  四大主流实验室的前沿模型（Claude/Anthropic、Codex-GPT/OpenAI、Grok/xAI、Gemini/Google）在同一任务集上解题，通过 **execution-based judge**（基于执行验证，如 unit test 或 stdin/stdout 检查）判断正误，排除主观评判偏见。

- **Collaboration Phase（合作阶段）**  
  所有教师通过验证的解决方案构成一个 **verifiable curriculum（可验证课程）**，并以两种方式用于训练学生模型：
  - **SFT（Supervised Fine-Tuning）**：将教师答案作为监督信号进行模仿。
  - **RLVR（Reinforcement Learning with Verifiable Rewards）**：将问题本身构建为 RL 环境，奖励 = 代码通过测试的比例。

### ⭐ 相比现有方法的优势
| 方面 | 本文改进 |
|------|--------|
| **教师评估公平性** | 使用 execution-based judge 替代 LLM-as-judge，消除 family bias 和 self-preference |
| **多教师整合方式** | 不是简单合并答案（pooling answers），而是联合构建一个可验证的学习环境（learning-by-doing） |
| **学生能力提升路径** | 发现 SFT 对已有竞争力的学生无效甚至有害，而 RLVR 能实现正向增益 |
| **可复现性与开源** | 完整发布代码、数据、测试用例、验证工具链及 on-prem 可运行管道（NVIDIA GB10），支持完全重现实验 |

---

## 2. 核心实验方法和设置

### 📚 数据集
| 类型 | 来源 | 描述 |
|------|------|------|
| **Function Tasks** | MBPP（HumanEval-style） | 共享教学 split，共 200 道函数编写题；MBPP-test 作为 held-out eval 集（150 题） |
| **Bug-Fix Tasks** | MBPP reference mutations | 通过对参考实现引入变异构造调试任务 |
| **Competition Problems** | DeepMind’s `code_contests` dataset | 高难度编程竞赛题（difficulty 6–9），共 150 题用于训练，另设 disjoint held-out set（68 题）用于评估 |

### 🔬 实验设置
- **Student Model**: Qwen2.5-Coder-7B（主）和 -32B（辅），采用 LoRA 微调
- **Teacher Models**: Claude, Codex (GPT), Grok, Gemini，均通过 headless CLI 接入
- **Shared Task Bank**: 所有教师面对相同问题，确保比较公平
- **Self-Correction Mechanism**: 若首次失败，返回错误信息允许最多两次重试，模拟教师修正过程
- **Intersection Control**: 学生训练集中仅保留所有教师都成功解决的问题（197 题），控制变量一致性
- **Evaluation Metric**: **pass@1（execution pass）**，即运行时是否通过隐藏测试

### 🧪 基线方法对比
| 方法 | 描述 |
|------|------|
| **Base** | 未经微调的学生模型 |
| **SFT (Single Teacher)** | 分别使用各教师验证后的解答进行监督微调 |
| **SFT (Union of All Teachers)** | 合并所有教师的答案进行 SFT |
| **RLVR (GRPO)** | 使用同一 curriculum 构建强化学习环境，使用 GRPO 进行训练，reward = 测试通过率 + 小格式奖励 |

---

## 3. 主要实验结果和性能指标

### 📊 教师竞争排名（Hard Competition Problems, pass@1）
| Teacher | Pass Rate | Code Extraction Success |
|--------|-----------|-------------------------|
| **Gemini** | **77% (115/150)** | 100% |
| **Claude** | 69% (104/150) | 96% |
| **Codex** | 69% (103/150) | 90% |
| **Grok** | 50% (75/150) | 73% |

> 💡 注：在简单任务（MBPP）中四者均接近饱和（99–100% after self-correction），无法区分能力差异；真正有效区分出现在高难度竞赛题上。

#### 公平性修正说明（Fairness Corrections）
- **Claude**: 初始中断于 122/150，补全后从 67% → 69%
- **Gemini**: 中途因 API 信用额度耗尽导致 34 题缺失，恢复后重新生成
- **Grok**: 批量调用存在 52% 空响应（timeout），加入 retry 机制后提取成功率从 48% → 73%

---

### 📉 SFT 结果：模仿学习对强学生有害
在 **intersection-controlled set（197 shared problems）** 上的结果显示：

#### Table: MBPP-test 上的 pass@1（7B & 32B 学生）
| Student | Base | +Claude | +Codex | +Grok |
|--------|------|--------|-------|------|
| **7B** | 76.7% | 74.0% | 70.0% | 69.3% |
| **32B** | 82.0% | 80.0% | 77.3% | 79.3% |

- 所有 SFT 版本均低于 base，且教师表现排序（Claude ≥ Codex > Grok）得以保留
- **Union of all teachers (SFT)**：MBPP 72.7%，competition 仅 **2.9%**（vs base 5.9%）

> ❗ 结论：**Imitation degrades competent coders** —— 已具备较强能力的学生通过模仿反而退步。

---

### 📈 RLVR 结果：可验证奖励显著提升性能
使用相同的 curriculum 作为 RLVR 环境（GRPO Trainer），结果逆转：

| Method | Competition Pass@1 (base → student) | Relative Gain |
|--------|-------------------------------|---------------|
| **SFT (union)** | 5.9% → **2.9%** | ↓ 降级 |
| **RLVR (GRPO, 200 steps)** | 5.9% → **7.4%** | ↑ +25% |
| **RLVR (GRPO, v2, 1000 steps)** | 5.9% → **8.8% peak** (at step 250–750) | ↑ **+49% relative gain** |

- 最终 step 1000 回落到 7.4%，建议选择中间 checkpoint
- 训练期间 reward 明显上升（~0 → 0.25–1.0），表明泛化能力增强
- 图表显示 held-out performance 与 training reward 正相关

> ✅ 关键洞见：**Same data, opposite direction** —— 数据相同，但 SFT 降低性能，RLVR 提升性能。

---

## 4. 关键结论和发现

### 🔑 主要发现
1. **Execution-based evaluation 是更公平的教师评估方式**  
   - 在高难度问题上可区分教师能力（Gemini > Claude ~ Codex > Grok）
   - 避免了 LLM-as-judge 的自偏好偏差

2. **Imitation (SFT) 不能提升已有竞争力的学生，反而可能损害性能**  
   - 即使是多个优秀教师的答案合并，也无法突破学生的“imitation ceiling”
   - 表明瓶颈不在知识容量，而在学习机制本身

3. **Verifiable Curriculum 作为 RLVR 环境能有效促进学生“做中学”**  
   - RLVR 成功将 base 5.9% 提升至峰值 8.8%（+49%）
   - 支持“价值不在于答案池，而在于构建可验证学习环境”的核心主张

4. **Collaborative curation > Answer pooling**  
   - 多教师的价值不是提供答案，而是共同构建高质量、可验证的任务集合

---

### ⚠️ 局限性（Limitations）
| 限制 | 说明 |
|------|------|
| **Benchmark Saturation** | MBPP 是公开数据集（2021），教师很可能已见过，影响难分真能力 |
| **Asymmetric Train-Leak Risk** | 无法确认各教师预训练语料中是否存在特定题目，可能导致不公平优势 |
| **Small Held-Out Set Size** | 竞赛测试集仅 68 题，±1 题即造成较大波动（sampling noise） |
| **Single Student Family** | 仅使用 Qwen2.5-Coder 系列，尚未验证其他架构 |
| **One Language Only** | 实验局限于 Python 编程任务 |
| **Hardware Instability** | GB10 平台出现 GPU GSP timeout（Xid 119/154），需额外容错设计 |

---

### 🔮 未来工作方向
1. **扩展到更多语言和任务类型**（如 JavaScript, C++, formal reasoning）
2. **动态 curriculum design**：根据学生表现自适应调整任务难度
3. **引入 human-in-the-loop verification** 以进一步提高 reward signal 质量
4. **探索 teacher specialization**：不同教师负责不同子领域的问题生成与验证
5. **构建 open benchmark for AI-teaching-AI pipelines**，推动该方向标准化评估

---

## ✅ 总结一句话
> **AI 教师协作的最大价值，不是让学生“模仿答案”，而是共同搭建一个“可验证的练习场”，让学生在“做题—反馈—优化”的闭环中真正成长。**

该研究揭示了当前多教师知识蒸馏范式的根本局限，并为下一代 AI 自我进化的教育框架提供了可复现、可验证的新路径。

</details>

---

### 16. [Scalable and Culturally Specific Stereotype Dataset Construction via Human-LLM Collaboration](https://arxiv.org/abs/2607.07895)

**Authors**: Weicheng Ma, John Guerrerio, Soroush Vosoughi  
**Category**: cs.CL  
**Published**: 2026-07-10  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.07895v1  

#### Abstract
Research on stereotypes in large language models (LLMs) has largely focused on English-speaking contexts, due to the lack of datasets in other languages and the high cost of manual annotation in underrepresented cultures. To address this gap, we introduce a cost-efficient human-LLM collaborative ann...

---

### 17. [Diarization-Guided Qwen-ASR Adaptation for Multilingual Two-Speaker Conversational Speech](https://arxiv.org/abs/2607.08208)

**Authors**: Hao Wu, RongQi Han, Zhen Wang, Wei Liang, Wei Xu  
**Category**: cs.CL  
**Published**: 2026-07-10  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.08208v1  

#### Abstract
This paper describes our self-designed system for Task 1 of the MLC-SLM 2026 Challenge for multilingual two-speaker conversational speech. The system combines a modular speaker diarization front end with a challenge-adapted Qwen3-ASR-1.7B recognizer. The diarization front end performs voice activity...

---

### 18. [It Takes a MAESTRO To Prune Bad Experts](https://arxiv.org/abs/2607.08601)

**Authors**: Palaash Goel, Ayush Maheshwari, Tanmoy Chakraborty  
**Category**: cs.CL  
**Published**: 2026-07-10  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.08601v1  

#### Abstract
Sparsely-activated Mixture-of-Experts (MoE) language models achieve remarkable inference efficiency by activating only a small fraction of parameters per token, yet their full expert banks reside in memory at all times, creating a prohibitive deployment bottleneck. Existing structured pruning method...

---

### 19. [Deep Learning for Joint Narrowband Interference Cancellation and Soft Demodulation in OFDM Systems](https://arxiv.org/abs/2607.08717)

**Authors**: Emmanouil Kavvousanos, Francky Catthoor, Vassilis Paliouras  
**Category**: cs.LG  
**Published**: 2026-07-10  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.08717v1  

#### Abstract
Narrowband interference (NBI) severely degrades orthogonal frequency-division multiplexing (OFDM) systems by corrupting subcarriers and rendering classical soft demodulation ineffective. Conventional compressed-sensing (CS) mitigation exhibits high sequential latency and leaves structured, non-Gauss...

---

### 20. [When LLMs Agree, Are They Right? Auditing Self-Consistency and Cross-Model Agreement as Confidence Signals](https://arxiv.org/abs/2607.08065)

**Authors**: Kaihua Ding  
**Category**: cs.AI  
**Published**: 2026-07-10  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.08065v1  

#### Abstract
LLM-as-judge (Zheng et al., 2023) is increasingly the default for evaluating AI systems in enterprise pipelines, often scaled to ensembles (Verga et al., 2024) or "mixture-of-experts" (Shazeer et al., 2017) panels of judges. These systems share a key assumption: that consistency -- agreement among j...

---

### 21. [Towards Precision Therapy in Hepatocellular Carcinoma: A Clinical-Reasoning LLM for Risk Stratification and Treatment Guidance](https://arxiv.org/abs/2607.08602)

**Authors**: Peng Cui, Jitao Wang, Siyan Xue, Yao Huang, Haoming Xia, Dong Li, Dengxiang Liu, Weilin Wang, Liping Liu, Leida Zhang, Yunfu Cui, Tao Peng, Daolin Ji, Haitao Zhao, Wei Zhang, Xiaojuan Wang, Weijie Ma, Zongren Ding, Jinlong Li, Yuan Ding, Jiajing Zhao, Zhiyu Chen, Chengkun Yang, Ziyue Huang, Jiaqi Liu, Fusheng Liu, Yang Zhou, Xiaojuan Wang, Zhongquan Sun, Shiyun Bao, Xiaojun Wang, Ming Yang, Guangxin Li, Bin Shu, Yong Liao, Hongxuan Li, Yao Tang, Shizhong Yang, Yongyi Zeng, Yufeng Yuan, Yinpeng Dong, Jihui Hao, Jun Zhu, Jiahong Dong  
**Category**: cs.AI  
**Published**: 2026-07-10  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.08602v1  

#### Abstract
Hepatocellular carcinoma (HCC) is a common malignancy and a leading cause of cancer-related mortality. Current guidelines and staging systems provide coarse categories, but often miss within-stage heterogeneity and the clinical context in electronic medical records (EMRs). We present HCC-STAR (Hepat...

---

### 22. [Tool-Making and Self-Evolving LLM Agents in Low-Latency Systems](https://arxiv.org/abs/2607.08010)

**Authors**: Kalle Kujanp\"a\"a, Ning Liu, Shahnawaz Alam, Yeshwanth Reddy Sura, Tianyu Yang, Kristina Klinkner, Shervin Malmasi  
**Category**: cs.CL  
**Published**: 2026-07-10  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.08010v1  

#### Abstract
Production LLM agents often waste latency and reliability by regenerating code for the same procedural steps on every request. We replace this inference-time coding loop with an agentic tool-making pipeline that compiles repeated SOP steps into validated, versioned tools before deployment. The tool-...

---

### 23. [Selective Left-Shift: Turning Test-Time Compute and Difficulty-based Curation into Training Data for Low-Resource Code Generation](https://arxiv.org/abs/2607.07748)

**Authors**: Didula Samaraweera, Anjana Supun, Srinath Perera  
**Category**: cs.LG  
**Published**: 2026-07-10  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.07748v1  

#### Abstract
Large Language Models achieve strong code generation for high resource languages like Python and Java but suffer sharp performance drops on Low-Resource Programming Languages~(LRPLs) such as Julia. Improving Small Language Models~(SLMs) for these languages faces a trilemma: Supervised Fine-Tuning~(S...

---

### 24. [Structure Learning on Clustered Data](https://arxiv.org/abs/2607.08238)

**Authors**: Ryan Thompson, Matt P. Wand, Veerabhadran Baladandayuthapani  
**Category**: cs.LG  
**Published**: 2026-07-10  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.08238v1  

#### Abstract
Recent algorithmic advances have made directed acyclic graph (DAG) structure learning scalable for causal discovery. Yet, the currently available techniques assume a completely homogeneous population, precluding their application to clustered data where cluster-specific variations (e.g., patient-spe...

---

### 25. [Write-Protected Discrete Bottlenecks for Language-Grounded World Models: A Structural Limitation and Sufficient Fix](https://arxiv.org/abs/2607.08312)

**Authors**: Jiayi Fang  
**Category**: cs.LG  
**Published**: 2026-07-10  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.08312v1  

#### Abstract
How should language interface with a world model's discrete symbol system? The dominant paradigm -- end-to-end injection of LLM/VLM features into robot world models (RT-2, Octo, PaLM-E) -- implicitly assumes that language gradients can directly shape physical symbol representations. We ask whether t...

---

### 26. [AutoAnchor: Stable Diffusion Unlearning Using Cross-Attention as a Manifold Surrogate](https://arxiv.org/abs/2607.08337)

**Authors**: Siyuan Wen, Jiahao Zeng, Ningning Ding  
**Category**: cs.LG  
**Published**: 2026-07-10  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.08337v1  

#### Abstract
Diffusion unlearning is essential for mitigating the generation of harmful or copyrighted content in text-to-image models. Current diffusion unlearning techniques determine the model update direction by either using alternatives of the target concept as an anchor or using empty prompts. The anchor-b...

---

### 27. [Context Graphs for Proactive Enterprise Agents](https://arxiv.org/abs/2607.07721)

**Authors**: Avinash Kumar  
**Category**: cs.AI  
**Published**: 2026-07-10  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.07721v1  

#### Abstract
Retrieval-Augmented Generation (RAG) and agentic frameworks have advanced enterprise AI considerably, yet agents remain fundamentally reactive: they wait for a human query before acting. This paper argues that genuine enterprise productivity gains require proactive agents: systems that surface relev...

---

### 28. [Agentic AI and Retrieval-Augmented Models in Straight-Through Underwriting](https://arxiv.org/abs/2607.07858)

**Authors**: Robert Richardson, Josh Meyers, Brian Hartman, David Sandberg  
**Category**: cs.AI  
**Published**: 2026-07-10  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.07858v1  

#### Abstract
Artificial intelligence (AI) is beginning to reshape actuarial practice, particularly in domains that require reasoning over unstructured documents, heterogeneous data sources, and regulated decision workflows. Actuaries now face a design space that ranges from traditional rule-based automation to l...

---

### 29. [Nigeria Machinery: A Low-Resource Industrial Dataset with a Domain-Grounded Reasoning Layer](https://arxiv.org/abs/2607.07883)

**Authors**: Gospel Bassey, Vincent Fakiyesi  
**Category**: cs.AI  
**Published**: 2026-07-10  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.07883v1  

#### Abstract
There is relatively little, public, and model-ready data on industrial machinery for African economies. This makes it hard to do quantitative analysis or to train language models on numeric tasks grounded in that setting. We release two things to help with part of this problem. The first is the Nige...

---

### 30. [Concretized Proposition Prompting Resolves Composition-Knowledge Dichotomy in Large Language Models](https://arxiv.org/abs/2607.08018)

**Authors**: Changhun Lee, Minguk Jeon, Jongkyung Shin, Chiehyeon Lim  
**Category**: cs.AI  
**Published**: 2026-07-10  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.08018v1  

#### Abstract
LLMs often struggle to balance compositionality with knowledgeability, a challenge we define as Composition-Knowledge Dichotomy. To address this, we propose Concretized Proposition Prompting (CPP), a framework that explicitly concretizes propositions relevant to questions. The results demonstrate th...

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

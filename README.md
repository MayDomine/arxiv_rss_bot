# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-08-07 06:59:19 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [PaDoc: Layout-Grounded Parallel Decoding for Document Parsing](https://arxiv.org/abs/2608.06146)

**Authors**: Hao Yu, Jiabo Zhan, Kang Liu, Linnan Zhao, Dongxu Yue, Rui Chen, Jinglin Wang, Chong Sun, Chen Li, Jing Lyu, Chun Yuan  
**Category**: cs.AI  
**Published**: 2026-08-07  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2608.06146v1  

#### Abstract
End-to-end document parsers provide a unified interface, but serialize page layouts and regional contents into one autoregressive sequence. This formulation forces independent regions onto a decoding path whose length grows with the total content, whereas crop-based two-stage parsers expose region-l...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《PaDoc: Layout-Grounded Parallel Decoding for Document Parsing》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的端到端（end-to-end）document parser 将页面布局和区域内容序列化为单一的自回归序列，导致独立区域之间存在不必要的顺序依赖，解码路径长度随总内容增长而线性增加，效率低下。另一方面，基于裁剪的两阶段方法（crop-based two-stage parsers）虽然实现了区域级并行，但需要重复进行视觉编码（visual pre-fill），且丢失了完整的页面上下文。

### 提出了什么新方法或新思路
本文提出 **PADoc**（Parallel Document Parser），一种**基于布局的并行解码框架**，其核心思想是：
- 将预测的布局视为共享页面表示上的分支结构（branching structure）。
- 在“区域充分性”（region-sufficiency）假设下，推导出一种**前缀条件分解**（prefix-conditioned factorization），使得布局流（layout stream）和各区域内容分支（content branches）可以并发推进。
- 利用 **packed variable-length ancestor attention** 实现训练时的标准 next-token 预测目标，同时保持正确的依赖关系。
- 推理时通过 **masked parallel decoding** 创建多个并发请求，后端 vLLM 利用缓存中驻留的共享前缀（cache-resident shared-prefix reuse）实现高效服务。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **上下文完整性** | 保留完整页面图像作为共享视觉前缀，避免两阶段方法中的上下文碎片化。 |
| **解码效率** | 解码深度从所有内容长度之和降低至最长的 layout-content 路径，显著减少关键路径长度。 |
| **系统实现** | 单一 MLLM 实现，无需额外检测头或识别头，训练与推理统一。 |
| **吞吐量与延迟** | 在相同骨干模型下，相比 Sequential SFT 基线，有效页吞吐提升 67.4–118%，P95 延迟降低 39.2–54.9%。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **OmniDocBench v1.6 Full**：共 1,651 页，涵盖多样化的文档类型、语言、布局和视觉条件，用于评估解析质量和布局分析。
- **384-page 子集**：从 OmniDocBench 中系统均匀采样得到，用于控制并发数下的推理效率评测。

### 实验设置和评估指标

#### （1）质量评估指标
| 类别 | 指标 |
|------|------|
| **Layout Analysis** | IoU, F1, Precision, Recall（按文本、图像、表格、公式分类）；Overall 宏平均；Full-page F1 |
| **End-to-end Parsing** |  
| - 文本 | Text Edit（越低越好）  
| - 公式 | Formula CDM（越高越好）  
| - 表格 | Table TEDS / TEDS-S  
| - 阅读顺序 | Read Order Edit  
| - 综合得分 | Overall（text + table + formula 的聚合得分）  

#### （2）效率评估设置
- **硬件平台**：单张 NVIDIA A800 80GB GPU
- **推理引擎**：vLLM 0.19.1 + FlashInfer backend
- **精度配置**：bfloat16
- **并发级别**：16, 32, 64, 128, 256 个并发文档请求
- **评估指标**：
  - 平均每秒有效页数（Mean valid pages per second per GPU）
  - P95 端到端延迟（P95 E2E latency）

#### （3）基线方法对比
| 类型 | 方法 |
|------|------|
| **Two-stage** | PaddleOCR-VL1.5 |
| **End-to-end** | Qianfan-OCR, DeepSeek-OCR-2, dots.ocr, HunyuanOCR-1.5, MonkeyOCRv2-B-Parsing, Sequential SFT（同骨干基线） |
| **Large VLMs** | Qwen3-VL-235B, Gemini 3 Pro/Flash, GPT-5.2, InternVL3.5-241B 等 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）布局分析性能（Table 1）
| 方法 | Overall F1 | 图像 F1 | 表格 F1 | 公式 F1 | Full-page F1 |
|------|------------|--------|--------|--------|--------------|
| **PADoc (Ours)** | **91.1** | 87.0 (=SOTA) | **97.0** | 93.7 | **88.1** |
- 在 end-to-end 方法中取得最高 Overall F1 和 Precision（93.3），尤其在表格类表现领先。

#### （2）端到端解析质量（Table 2）
| 方法 | Overall ↑ | Text Edit ↓ | Formula CDM ↑ | Table TEDS ↑ |
|------|----------|-------------|----------------|---------------|
| **PADoc (Ours)** | **94.24** | **0.038** | **95.59** | 90.94 |
| Qianfan-OCR | 93.90 | 0.040 | 95.08 | 90.53 |
| HunyuanOCR-1.5 | 94.74 | 0.039 | 94.50 | 93.67 |
- 在 end-to-end 方法中排名前列，**Text Edit 和 Formula CDM 均达最优**。
- 虽然 Overall 略低于 HunyuanOCR-1.5，但在更小规模（2.1B vs 1.0B）下实现接近甚至超越的表现。

#### （3）推理效率（Table 3）
| 方法 | 吞吐量（pages/s/GPU）↑ | P95 Latency (s) ↓ |
|------|------------------------|--------------------|
| **PADoc @ C64** | **1.722** (+110% vs Seq SFT) | **70.659s** (-49.9% vs Seq SFT) |
| Sequential SFT (baseline) | 0.829 | 141.038s |
- 在所有并发等级下均为**最快的 end-to-end parser**。
- 吞吐提升 **67.4–118%**，P95 延迟降低 **39.2–54.9%**。
- 即使参数量更大（2.1B），仍优于 1.0B 的 HunyuanOCR-1.5 和 0.7B 的 MonkeyOCRv2。

#### （4）消融实验与理论加速分析（Appendix D.2）
- **Forward Step Reduction**：
  - 序列化解码平均步数：923.2
  - PADoc 关键路径步数：325.3
  - **中位数减少 58.4%**（IQR: 47.7–68.2%）
- **实测端到端加速比**：
  - 中位加速比 1.453× ~ 1.937×（对应延迟下降 31.2–48.4%）

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **布局引导的并行解码可行且高效**：通过将布局结构建模为分支图，可在不牺牲全局上下文的前提下实现区域内容的并发生成。
2. ✅ **训练与推理一致性高**：利用 packed variable-length ancestor attention 可在标准 next-token 训练范式中隐式学习并行结构，无需辅助损失或多模块设计。
3. ✅ **KV Cache 复用极大提升效率**：共享前缀（image + layout prefix）在多个 content stream 间自动复用，大幅减少重复计算和内存占用。
4. ✅ **质量与效率双赢**：尽管采用并行策略，PADoc 在 layout F1 和 end-to-end parsing 指标上仍达到 SOTA 水平，同时显著优于同类方法的推理性能。

### 方法的局限性
- **依赖 layout prediction 准确性**：若初始 layout 流预测错误，后续所有 content branches 将继承该错误，缺乏纠错机制。
- **动态分支管理复杂度高**：需调度器支持动态创建 content stream，对 serving runtime 有一定要求。
- **长文档分支数量受限**：最多支持 255 个 content branches，可能限制极复杂文档的应用。
- **未支持跨区域交互**：当前假设各区域内容条件独立，忽略了某些语义关联（如跨表引用、脚注等）。

### 未来工作方向
- 引入轻量级 feedback mechanism，在 content decoding 过程中微调 layout prediction。
- 扩展为 hierarchical layout parsing，支持嵌套结构（如 section → paragraph → formula）。
- 结合 speculative decoding 或 cascaded attention 进一步优化短分支的启动开销。
- 探索在多页文档或文档集合上的全局 context sharing。

---

> 🔗 **代码地址**：https://github.com/Longin-Yu/Padoc  
> 📄 **论文原文**：https://arxiv.org/abs/2608.06146

</details>

---

### 2. [Routing LLM Inference to the Cleanest Grid in Real Time](https://arxiv.org/abs/2608.06188)

**Authors**: Aleks Bernhard, Arif Baran Yardimci  
**Category**: cs.DC  
**Published**: 2026-08-07  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.06188v1  

#### Abstract
Large-language-model inference is a fast-growing electricity load whose marginal carbon intensity varies by more than an order of magnitude across grid regions and across the day, making request placement an attractive lever: no retraining, no hardware change. We report a live validation of carbon-a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Routing LLM Inference to the Cleanest Grid in Real Time

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLM）推理已成为云计算中快速增长的电力负载，其碳排放强度在不同电网区域和时段差异巨大（超过一个数量级）。如何在不改变模型或硬件的前提下，通过调度决策降低推理任务的碳足迹，是一个高价值且低摩擦的优化路径。

本文聚焦于 **carbon-aware inference routing** ——即根据实时边际碳排放信号（MOER），将 LLM 推理请求动态路由到“最清洁”的电网区域，从而减少整体运营碳排放。

---

### 提出的新方法与创新思路

作者提出并验证了一种 **基于 MOER 信号的实时推理路由机制**，作为生产级负载均衡器上的可逆叠加层（reversible overlay）。该方法具备以下四项关键创新：

1. ✅ **真实生产环境下的可行性验证**  
   首次在真实的多区域 GPU 测试平台上实现了 **live carbon-aware routing**，使用实际的生产压力路由器（production pressure router）作为基线，而非理想化的 round-robin 或 uniform 路由。

2. ✅ **基于 GPU telemetry 的精细化能耗建模**  
   使用 NVIDIA DCGM 工具采集 GPU 功耗数据，构建并发度（concurrency）与每输出 token 能耗的关系曲线（Phase-0 energy characterization），避免依赖粗略的 nameplate TDP 估算。

3. ✅ **post-hoc carbon settlement 方法论**  
   所有碳排放结算均基于 **历史 MOER 数据**（historical MOER），而非仅依赖路由时的预测值。这确保了碳核算的真实性和可审计性，形成“预测驱动决策，历史完成结算”的闭环。

4. ✅ **跨区域比较中的信号选择指导原则**  
   明确指出：**应使用绝对 MOER（lbs CO₂/MWh）而非百分位 signal-index 进行跨区域比较**。后者是为单区域时间维度调度设计的，用于空间决策会导致错误排序（例如将更脏的区域误判为更清洁）。

---

### 相比现有方法的优势

| 维度 | 本文方法 | 先前工作典型做法 |
|------|--------|----------------|
| **部署真实性** | Live 多区域 GPU 部署，无调度失败 | 多为仿真或回放（replay/simulation） |
| **控制基线** | 使用真实生产压力路由器（off arm） | 常用 round-robin 或 uniform 分配 |
| **能耗建模** | 基于 DCGM telemetry 的并发感知能耗模型 | Nameplate TDP 或静态估计 |
| **碳核算方式** | 决策用 forecast MOER，结算用 historical MOER | 通常只用 forecast，未区分 |
| **信号解释** | 强调跨区选绝对 MOER，非 signal-index | 缺乏对此陷阱的实证说明 |

> 🔍 **一句话优势总结**：这是首个在真实生产架构上端到端验证可行、并提供可信碳核算的方法，同时揭示了一个关键实践误区（signal-index 误用）。

---

## 2. 核心实验方法和设置

### 使用的数据集与信号源

- **MOER 数据来源**：WattTime V3 API  
  - `co2_moer` 实时与历史接口
  - 模型版本：North-America MOER model 2026-03-01（增强对 renewable curtailment 的检测）
  - 时间分辨率：5分钟（实时）、小时级（回放）
- **能源数据**：来自 Phase-0 并发扫描实验，使用 NVIDIA DCGM 收集功耗
- **推理模型**：Llama-3.1-8B-Instruct on vLLM
- **测试平台**：Solyx AI Grid，支持跨区域调度

---

### 实验设置

#### （1）**Live Two-Region Testbeds**

| 测试组 | GPU 类型 | 区域对 | 持续时间 |
|-------|--------|------|--------|
| A100 | A100-SXM4-40GB | AZPS（亚利桑那，太阳能丰富） vs PJM-DC（弗吉尼亚，化石燃料为主） | ~48 小时 |
| H100 | H100-SXM5-80GB | ERCOT-NC（德州达拉斯，风光波动大） vs SOCO（佐治亚，窄幅高碳） | 提前终止（因信号异常） |

- **工作负载**：合成开放循环流，5 req/s，共约 1.96M 请求轮次
- **会话类型**：40% 单轮、50% 多轮、10% 长上下文 RAG
- **路由策略（arms）**：
  - `round-robin`
  - `off`（生产压力路由器，盲基线）
  - `eco-low`, `eco-med`, `eco-high`（碳权重 w=0.25/0.5/1.0）

> ⚠️ 注意：所有 arms 共享物理资源（GPU、队列），存在干扰效应（SUTVA violation），因此延迟结果为观察性而非因果估计。

#### （2）**Historical MOER Replay（主定量分析）**

- **时间跨度**：2025-07-06 至 2026-07-06（完整一年，8,760 小时）
- **地理范围**：19 个 CONUS grid regions（如 CAISO, ERCOT, PJM-Chicago 等）
- **容量假设**：每个区域最多承载 50% 总需求（cap 限制集中程度）
- **能量建模**：固定为 H100 饱和并发下的 ~0.104 J/token
- **session pinning**：40% 多轮请求绑定至起始区域（模拟对话连续性）
- **评估指标**：
  - 每请求归因碳排放（gCO₂/request）
  - 相对于 round-robin 和碳盲负载均衡器的减排比例
  - Block-bootstrap CI（5天块，10,000 次重采样）

---

### 基线方法对比

| 基线 | 描述 |
|-----|------|
| `round-robin` | 均匀分配请求，理论公平基线 |
| `off`（生产压力路由器） | 实际生产中使用的基于队列/负载的压力加权路由 |
| `carbon-blind load balancer`（回放中建模） | 类似“本地优先，溢出则转发”策略，用于验证基线鲁棒性 |

---

## 3. 主要实验结果和性能指标

### Live 实验结果（定性验证可行性）

| 指标 | 结果 |
|------|------|
| **系统稳定性** | 无任何 dispatch failure，运行 48 小时以上 |
| **碳减排方向** | 在 A100 测试中，`eco-high` 比 `off` 减排 **1.45%**（方向正确） |
| **p95 延迟影响** | `off`: 18.7s → `eco-high`: 20.9s（↑11.7%，受共享资源干扰，仅供参考） |
| **流量迁移效果** | 随着碳权重增加，清洁区域（AZPS）流量占比从 50%（round-robin）升至 63%（eco-high），单调递增 |

> 📌 **结论**：机制可行，能有效引导负载向清洁区域迁移，但短期双节点实验无法得出普适减排幅度。

---

### Historical Replay 结果（定量潜力评估）

| 配置 | 减排幅度（vs round-robin） | 说明 |
|------|----------------------------|------|
| **Primary Configuration** | **50.9%**（95% CI: 48.5–53.3%） | 容量受限的 cleanest-first 调度 + session pinning |
| vs modeled carbon-blind balancer | **51.0%** | 表明基线选择不影响核心结论 |
| Soft multiplier policy（模拟 live overlay） | **~23%** | 更保守的操作策略下仍显著 |
| Aggressive policies（cleanest-first + α 变化） | **34–51%** | 取决于对零 MOER 区间的信用处理方式 |

#### 关键分解（Figure 4）

- **静态年均 MOER 策略**（static annual-mean MOER）贡献约 **31.6 个百分点**（~60% 的总潜力）
- **动态实时 MOER 调整**（hourly routing）额外贡献 **~22.4 个百分点**（约 40%）
- **Session pinning 开销**：使理想 54.0% 下降至 50.9%，损失约 3.1 个百分点

> 💡 即：**实时信号本身贡献了近 40% 的减排收益**，说明高频调度具有重要价值。

---

### 消融实验与敏感性分析

| 因素 | 影响 |
|------|------|
| **是否保留 zero-MOER intervals** | 若删除这些时刻，减排从 51% ↓ 至 24.5% |
| **zero-MOER 信用赋值（α-sweep）** | 当 α=1（视为无减排价值），仍可实现 **~34%** 减排 |
| **季节性变化** | 春季（MAM）可达 63.7%，日间峰值（太阳能高峰）达 69.4% |
| **block-bootstrap 块长度** | CI 随块增大略有扩大（1天→14天），但点估计稳定 |

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **Carbon-aware routing 是可行的**  
   在真实生产级多区域 GPU 架构上，以 MOER 为信号的路由策略可以安全、可逆地集成，并成功引导负载迁移。

2. ✅ **减排潜力巨大（理论上限）**  
   在理想条件下（完美预测、充足容量、session pinning 存在），**GPU 层面的 operational emissions 最多可减少 50.9%**。

3. ✅ **实时信号至关重要**  
   动态响应 hourly MOER 变化（尤其是 renewable curtailment）贡献了约 **40% 的减排效果**，远超静态政策。

4. ❗ **跨区域比较必须使用绝对 MOER，而非 signal-index**  
   否则可能因区域分布差异导致反向决策（把更脏的区域当成更干净的）。这是一个已被忽视但极具实践意义的陷阱。

5. ⚠️ **利用率与能效交互作用不可忽略**  
   GPU 能耗随并发度提升下降达 ~30×（H100 从 3.37 → 0.104 J/token），意味着一个高利用率的“较脏”区域可能比低利用率的“清洁”区域更环保。

---

### 方法的局限性

| 局限 | 说明 |
|------|------|
| **非真实预测调度** | Replay 使用 historical MOER，代表“完美预知”，实际部署会因 forecast error 而打折 |
| **未建模跨区延迟与数据传输成本** | 忽略了网络开销、prefix-cache 失效等问题 |
| **未考虑 PUE 与设施层面碳排放** | 仅计算 GPU 直接能耗 × MOER，未包含冷却等设施开销（PUE） |
| **session pinning 抑制了部分潜力** | 40% 请求被锁定，限制了最大调度灵活性 |
| **缺乏 prefix caching 建模** | 跨区 dispatch 可能导致 APC/RadixAttention 缓存失效，增加 prefill 成本 |

---

### 未来工作方向

1. **加入物理容量约束（headroom modeling）**  
   基于实际可吸收的 curtailed renewable MW 数量，建立 `savings = f(headroom, α)` 曲面。

2. **开展多季节、深消纳场景的 live 验证**  
   如 CAISO / ERCOT 对比，在真实高波动性环境下连接 feasibility 与 magnitude。

3. **完善操作最佳实践（operational best practices）**  
   - 当 MOER 未知时，应禁用碳项并回退到压力路由，而非默认“视为清洁”
   - 将并发能效曲线与 prefix-cache 局部性纳入调度决策

4. **推动行业指南制定**  
   与 WattTime 合作发布指导文档：**跨区选绝对 MOER，区内选 signal-index**

5. **引入 real-time PUE 模型**  
   结合气候与冷却效率，实现全设施层级的碳感知调度。

---

> ✅ **最终结论**：  
> 本文证明了 **routing LLM inference to the cleanest grid 是一项技术上可行、碳减排潜力巨大的杠杆**。虽然 live 实验仅展示方向正确性，但一年期的历史回放表明，在合理假设下，**GPU 层面的碳排放可减少约 50%**。真正的挑战不在技术，而在 **信号理解、系统集成与运营规范** ——而这正是本文最重要的实践贡献。

</details>

---

### 3. [MicroEvo: Knowledge-Guided LLM Sampling for Efficient Microarchitecture Design Space Exploration](https://arxiv.org/abs/2608.06183)

**Authors**: Jia Xiong, Runkai Li, Chenxu Niu, Guangyuan Gao, Changwen Xing, Yifan Zhang, Xinlai Wan, Jieran Cui, Chen Bai, Yusheng Hua, Ying Wang, Ming Ling, Xi Wang, Tao Xie  
**Category**: cs.AI  
**Published**: 2026-08-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.06183v1  

#### Abstract
Microarchitecture design space exploration suffers from expansive search spaces and expensive PPA evaluation, leaving only a small simulation budget for design decision-making. Existing methods perform blind search without considering microarchitectural dependencies and fail to learn from the iterat...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：MicroEvo: Knowledge-Guided LLM Sampling for Efficient Microarchitecture Design Space Exploration**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现代微架构设计空间探索（Microarchitecture Design Space Exploration, DSE）面临两大挑战：
- **搜索空间巨大**：微架构参数组合呈指数级增长（如文中示例达 $3.95 \times 10^{13}$ 种配置），导致穷举不可行。
- **PPA评估成本高昂**：每次性能（Performance）、功耗（Power）、面积（Area）评估依赖周期精确仿真（如GEM5）和功耗建模（如McPAT），耗时长（单次可达数分钟至小时级），严重限制可用评估次数。

现有方法（如NSGA-II、BOOM-Explorer等）存在以下不足：
- **盲目搜索**：缺乏对微架构组件间依赖关系的理解，易陷入局部最优。
- **反馈利用低效**：无法从历史评估中提取可复用的知识指导后续搜索。
- **采样效率低下**：在有限预算下难以逼近高质量Pareto前沿。

### **提出的新方法与创新思路**
作者提出 **MicroEvo** —— 一种结合大语言模型（LLM）与蒙特卡洛树搜索（MCTS）的知识引导型多目标优化框架，其核心创新包括：

#### ✅ **1. LLM驱动的进化算子（LLM-Driven Evolutionary Operators）**
- 利用预训练LLM内部蕴含的微架构知识（来自教科书、设计文档等），生成符合架构约束和组件依赖的高质量初始设计。
- 设计两种算子：
  - **Knowledge Tuner**：基于全局优化知识进行局部精细化调整。
  - **Pattern Explorer**：分析兄弟节点间的模式差异，提出结构性创新方案。

#### ✅ **2. Pareto-UCT 决策策略**
- 改进标准UCT公式，引入多目标优化机制：
  - **Hypervolume Improvement (HVI)**：衡量候选节点对Pareto前沿的边际贡献。
  - **Crowding Distance**：鼓励探索稀疏区域，提升解的多样性。
  - 动态衰减的探索系数 $\lambda$ 平衡早期探索与后期开发。

#### ✅ **3. 主动知识积累机制（Active Knowledge Accumulation, AKA）**
- 从历史探索中自动提取可复用的优化知识：
  - **Pareto Analysis**：识别当前Pareto最优集合，分析覆盖缺口，指导搜索方向。
  - **Pairwise Analysis**：比较父子节点PPA变化，提炼具体调参规则（如“增大ROB需同步增加Load Queue”）。
- 构建**效用感知记忆库（Utility-Aware Memory）**，按效用分数检索最相关知识，避免重复使用无效规则。

#### ✅ **4. 状态感知指令机制（State-Aware Directive, SAD）**
- 实时监控搜索状态（停滞次数、局部增益），动态切换三种模式：
  - `exploit`：聚焦局部改进；
  - `balance`：协调结构优化；
  - `explore`：鼓励大跨度结构调整。
- 通过自然语言指令引导LLM行为，实现自适应搜索。

---

## **2. 核心实验方法和设置**

### **使用的数据集与平台**
- **主测试平台**：基于RISC-V的乱序执行核（Alpha21264风格）
  - 工具链：GEM5（cycle-accurate simulation） + McPAT（power/area modeling）
  - 设计空间：共22个参数，涵盖前端、执行单元、内存系统等，总配置数约 $3.95 \times 10^{13}$
- **扩展验证平台**：更复杂的工业级开源处理器 **XiangShan Kunminghu**
  - 特征：TAGE分支预测器、多级硬件预取器，设计空间更大（$\sim 3.41 \times 10^{26}$）

### **实验设置**
- **评估预算**：20轮或45轮PPA评估（对应5次初始化 + 10轮迭代）
- **LLM模型**：DeepSeek-V3.2 和 Gemini-3-pro
- **每轮操作**：
  - MCTS选择叶节点 → LLM生成2个新设计（每个算子各1个）
  - 每轮注入3条最高效用的知识提示
- **运行环境**：Intel Xeon Platinum 8480+服务器，5次独立实验取平均值

### **评估指标**
| 指标 | 定义 | 含义 |
|------|------|------|
| **Hypervolume (HV)** | 被Pareto解集支配的目标空间体积 | 数值越大越好，反映收敛性与分布性 |
| **ADRS (Average Distance to Reference Set)** | 当前解集到参考Pareto前沿的平均距离 | 数值越小越好 |
| **Search Efficiency (N_t)** | 达成目标HV所需的最少评估次数 | 数值越小，效率越高 |

### **基线方法对比**
| 方法 | 类型 | 说明 |
|------|------|------|
| **NSGA-II** | 进化算法 | 经典多目标遗传算法 |
| **MOTPE** | 贝叶斯优化 | 多目标Tree-structured Parzen Estimator |
| **BOOM-Explorer** | 贝叶斯优化 | 基于聚类特征加速探索 |
| **RL-DSE** | 强化学习 | 学习组件级参数选择策略 |
| **LEMOE** | LLM驱动 | 当前最先进的LLM+贝叶斯优化方法 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见Table 1）**

| 方法 | HV (Budget=45) ↑ | HV Improvement vs NSGA-II | ADRS ↓ | Search Efficiency (vs NSGA-II) |
|------|------------------|----------------------------|--------|-------------------------------|
| **NSGA-II** | 0.817 | — | 0.0524 | 1× |
| **LEMOE** | 0.679 | +2.53% | 0.0498 | — |
| **Boom-Explorer** | 0.709 | +7.0% | 0.0537 | — |
| **RL-DSE** | 0.711 | +7.36% | 0.0526 | — |
| **MicroEvo (DeepSeek)** | **0.817** | **+23.35%** | **0.0521** | **10.6×** |
| **MicroEvo (Gemini)** | **0.821** | **+23.94%** | **0.0523** | **10.6×** |

> 📌 **最大提升**：相比NSGA-II，HV提升高达 **36.2%**（以HV=0.74为阈值计算），搜索效率提高 **10.6倍**。

### **与基线方法的对比结果**
- **Pareto质量显著领先**：
  - 在相同评估次数下，MicroEvo获得更广、更凹的Pareto前沿（图8）。
  - 即使在仅20次评估的小预算场景下，仍能发现更多优质解。
- **收敛速度更快**：
  - 图9显示，MicroEvo仅需约 **70次评估** 即可达到NSGA-II需要 **750次以上** 才能达到的HV水平。
- **优于其他LLM方法**：
  - 尽管LEMOE也使用LLM，但因缺乏知识积累与状态控制，后期易陷入停滞；而MicroEvo持续进化，表现稳定上升（图10）。

### **消融实验结果（Table 3）**
移除任一组件均导致性能下降，验证各模块必要性：

| 消融配置 | HV Drop (Budget=45) | ADRS Increase |
|---------|--------------------|-------------|
| w/o AKA, SAD | -6.73% | +52.49% |
| w/o Tuner | -6.98% | +44.60% |
| w/o Explorer | -8.63% | +57.81% |
| w/o Pareto Analysis | -3.75% | +23.99% |
| w/o Pairwise Analysis | -5.12% | +35.39% |
| w/o Utility Memory | -4.01% | +22.05% |
| w/o SAD | -5.34% | +35.86% |

> 🔍 **关键发现**：Pairwise Analysis 对性能影响最大，表明从具体案例中提取细粒度调参经验至关重要。

---

## **4. 关键结论和发现**

### **主要发现**
1. **LLM不仅是生成器，更是推理引擎**：  
   MicroEvo成功将LLM从“黑盒采样器”转变为“知识驱动的智能决策者”，通过prompt工程将其推理能力融入整个DSE闭环。

2. **知识积累是高效搜索的关键**：  
   单纯依赖LLM先验知识不足以支撑长期优化，必须结合**主动学习机制（AKA）**，实现“边探索、边总结、边应用”。

3. **动态调节搜索行为至关重要**：  
   固定策略（如始终exploit）会导致早熟收敛，SAD机制实现了**exploitation与exploration的在线平衡**。

4. **方法具有强可扩展性**：  
   在复杂工业级XiangShan处理器上，MicroEvo依然保持领先，并超越人工设计（图15）：
   - 几何平均能效提升 **8.6%**
   - 在`mm`, `rsort`, `spmv`等关键负载上优势明显

### **局限性**
- **依赖LLM接口稳定性**：若LLM响应延迟高或API不稳定，可能影响整体流程。
- **知识提取准确性受限于LLM理解能力**：错误归纳可能导致误导性建议。
- **当前未支持跨工作负载迁移**：所有知识均在单一workload下积累，尚未验证跨任务泛化能力。

### **未来工作方向**
- 探索 **offline知识蒸馏**，将积累的经验固化为轻量级模型，减少对LLM API的依赖。
- 构建 **跨基准测试的知识迁移机制**，提升框架通用性。
- 结合 **hardware-in-the-loop feedback**，进一步缩小仿真与实测之间的gap。
- 扩展至 **High-Level Synthesis (HLS)** 或 **chip floorplanning** 等EDA领域。

---

> ✅ **总结一句话**：  
> **MicroEvo首次实现了LLM在微架构DSE中的“自我进化”——不仅靠知识起跑快，更能通过持续反思与自适应调控，跑得远、跑得准。**

</details>

---

### 4. [Alternating Levenberg-Marquardt Training of Physics-Informed Neural Networks with Fourier-Enhanced Features](https://arxiv.org/abs/2608.05892)

**Authors**: Yulun Wu, Matthieu Barreau, Miguel Aguiar, Karl H. Johansson  
**Category**: cs.LG  
**Published**: 2026-08-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.05892v1  

#### Abstract
Physics-informed neural networks (PINNs) often fail to accurately resolve partial differential equations (PDEs) with high-frequency or multi-scale solutions, as well as strongly nonlinear problems. Two factors underlie this difficulty: spectral bias, the tendency of neural networks to underfit high-...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Alternating Levenberg-Marquardt Training of Physics-Informed Neural Networks with Fourier-Enhanced Features*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

该论文针对 **Physics-Informed Neural Networks (PINNs)** 在求解高频率、多尺度或多分量非线性偏微分方程（PDEs）时表现不佳的问题，提出了系统性的解决方案。其核心挑战包括：

- **Spectral Bias**：神经网络倾向于优先学习低频特征，难以捕捉高频或快速振荡的解。
- **Representation-Coefficient Coupling**：在标准 PINN 中，隐层参数（表示学习）与输出层系数（投影拟合）在同一个非凸优化目标中联合训练，导致优化困难、收敛慢且不稳定。

### 提出了什么新方法或新思路

作者提出了一种名为 **FALM-PINN**（Fourier-enhanced Alternating Levenberg-Marquardt PINN）的新框架，其核心思想是**将表示学习与系数拟合解耦**，通过交替优化实现更高效稳定的训练。

#### 主要创新点：

1. **双层交替优化架构**：
   - **上层问题（Upper-level）**：更新隐层参数 $ w $，生成一个由 **Fourier Feature Mapping** 增强的自适应基函数 $ \phi_d(x) $，从而丰富潜在空间中的高频成分。
   - **下层问题（Lower-level）**：固定基函数后，使用 **Levenberg-Marquardt (LM) 算法** 求解投影系数 $ \beta $，将其建模为一个非线性最小二乘问题。

2. **引入 LM 算法处理非线性 PDE**：
   - 下层问题利用 LM 算法对残差进行一阶线性化，每一步求解一个带阻尼项的严格凸子问题，具有闭式解。
   - 对于线性 PDE，该过程退化为单步 Kernel Ridge Regression (KRR)，保证全局最优；对于非线性 PDE，仍能稳定逼近局部最优。

3. **理论保障**：
   - 证明了所提出的交替优化算法在一般非线性 PDE 上的**全局收敛性**（Theorem 1），即使下层问题是非凸的。
   - 将 Fourier Feature 映射到隐空间解释为一种 **Deep Kernel Learning**，从理论上说明其缓解 spectral bias 的机制（Lemma 1）。

### 相比现有方法的优势

| 方面 | FALM-PINN 的优势 |
|------|------------------|
| **精度** | 在多个高频率、强非线性 PDE 上，相对 $ L_2 $ 误差比现有 SOTA 方法低 **1–2 个数量级**。 |
| **稳定性** | 交替优化 + LM 更新显著提升训练稳定性，避免陷入不良局部极小值。 |
| **通用性** | 可处理**标量/向量、线性/非线性、耦合系统**等广泛类型的 PDE，而如 IFeF-PINN 仅适用于线性标量 PDE。 |
| **理论支持** | 提供了完整的收敛性分析，优于大多数经验性改进方法。 |

---

## 2. 核心实验方法和设置

### 使用的数据集（PDE 测试基准）

论文在多个经典且具有挑战性的 PDE 上进行了验证：

1. **2D Klein-Gordon Equation**：非线性波动方程，测试长时间演化能力。
2. **1D Korteweg-de Vries (KdV) Equation**：含三阶导数的色散波方程，测试非线性传播结构。
3. **1D Heat Equation with High-Frequency Solution**：$ u(t,x) = e^{-t}\sin(F\pi x) $，其中 $ F=100 $，专门用于测试 spectral bias 缓解能力。
4. **Lid-driven Cavity Flow**：二维稳态不可压缩 Navier-Stokes 方程，测试多变量耦合场（速度 $ u_1, u_2 $ 和压力 $ p $）的建模能力。
5. **1D Viscous Burgers Equation**：具有激波结构的非线性对流扩散方程，测试陡梯度区域的逼近能力。

### 实验设置和评估指标

- **评估指标**：
  - **Relative $ L_2 $ Error**：
    $$
    \text{Relative } L_2 \text{ error} = \frac{\sqrt{\sum_k (u(x_k) - \hat{u}(x_k))^2}}{\sqrt{\sum_k u(x_k)^2}}
    $$
  - 报告五次独立运行的均值 ± 标准差。
  - 同时记录训练时间（GPU 时间/千迭代）和峰值显存占用。

- **训练设置**：
  - 使用 **Adam** 优化器。
  - 采用两阶段训练策略：
    1. **Warm-up Phase**（前 5,000 或 10,000 步）：所有参数联合训练以获得良好初始化。
    2. **Alternating Phase**：交替执行上层（更新 $ w $）和下层（LM 求解 $ \beta $）优化。
  - Fourier Feature 维度 $ D = 800 $，带宽 $ \sigma $ 根据任务调整（如 $ \sigma=10 $ 用于高频热方程）。

### 基线方法对比

| 方法 | 类型 | 关键技术 |
|------|------|----------|
| **Vanilla PINN** | 基线 | 标准 MLP + Adam |
| **RBA** [31] | 损失加权 | 残差驱动的注意力权重 |
| **CompleX-PINN** [40] | 激活函数 | 可学习 Cauchy 激活函数 |
| **PIKAN** [43] | 架构修改 | Kolmogorov-Arnold 网络 |
| **SIREN** [39] | 激活函数 | Sinusoidal 激活函数 |
| **IFeF-PINN** [52] | 解耦方法 | 本文方法的前身，仅适用于线性 PDE |

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

| PDE 任务 | 最佳 Baseline 相对 $ L_2 $ 错误 | **FALM-PINN** 相对 $ L_2 $ 错误 | **提升倍数** |
|--------|-------------------------------|-------------------------------|-------------|
| 2D Klein-Gordon | $ 1.14\times10^{-3} $ (PIKAN) | $ \mathbf{3.50\times10^{-5}} $ | ~33× |
| 1D KdV | $ 7.38\times10^{-3} $ (RBA) | $ \mathbf{4.42\times10^{-4}} $ | ~17× |
| 1D Heat ($ F=100 $) | $ 1.0\times10^{-1} $ (SIREN) | $ \mathbf{6.8\times10^{-4}} $ | >100× |
| Lid-driven Cavity | $ 5.07\times10^{-3} $ (SIREN) | $ \mathbf{2.78\times10^{-3}} $ | ~1.8× |
| 1D Viscous Burgers | $ 1.53\times10^{-2} $ (SIREN) | $ \mathbf{5.10\times10^{-5}} $ | ~300× |

> ✅ 所有任务中，**FALM-PINN 均取得最低误差**，尤其在高频率和强非线性任务上优势巨大。

### 与 IFeF-PINN 的直接对比（Table 6）

| Benchmark | IFeF-PINN | **FALM-PINN** |
|---------|-----------|---------------|
| 2D Klein-Gordon | $ 1.28\times10^{-3} $ | $ \mathbf{3.50\times10^{-5}} $ |
| 1D KdV | $ 1.32\times10^{-2} $ | $ \mathbf{4.42\times10^{-4}} $ |
| 1D Viscous Burgers | $ 2.46\times10^{-3} $ | $ \mathbf{5.10\times10^{-5}} $ |

> 🔍 表明：**使用 LM 求解器比使用 L-BFGS 更有效**，即使共享相同的 Fourier-enhanced 基函数，FALM-PINN 也因利用了最小二乘结构而显著更优。

### 消融实验结果（Ablation Study）

- **Fourier Feature 带宽 $ \sigma $ 的影响**（Table 3）：
  - 当空间频率 $ F $ 较高（如 $ F=200 $）时，必须使用较大的 $ \sigma $（如 $ \sigma=50 $）才能准确重建解。
  - 若 $ \sigma $ 过小（如 $ \sigma=1 $），模型无法捕获高频模式，预测失败。
  - 验证了 **Lemma 1** 的理论预测：带宽应与目标解的主导频率匹配。

- **两阶段训练必要性**：
  - 图中显示，在 warm-up 阶段结束后切换至 alternating 优化，误差出现“断崖式”下降。
  - 表明良好的初始化对 LM 收敛至关重要。

---

## 4. 关键结论和发现

### 主要发现

1. **解耦优化显著提升性能**：将 basis learning 与 coefficient fitting 分离，并分别用专用优化器处理，可极大缓解 PINN 的训练困难。
2. **Fourier-enhanced latent space 能有效对抗 spectral bias**：将 Fourier Feature 应用于隐层而非原始输入，结合 adaptive kernel learning 视角，提供了更强的高频表示能力。
3. **LM 算法优于通用优化器**：在下层问题中利用残差的最小二乘结构，通过 LM 算法实现快速、稳定的系数更新，远胜于 L-BFGS 等通用方法。
4. **理论与实践一致**：提出的交替算法被证明具有全局收敛性，且实验结果验证了其在复杂 PDE 上的鲁棒性和高精度。

### 方法的局限性

- **计算与内存开销较高**：
  - LM 算法需要构建和求解 Jacobian 矩阵 $ J^TJ + \gamma I $，其维度随采样点数增长，导致**时间和显存成本高于大多数基线**（见 Table 5）。
  - 不适合超大规模网格或极高维 PDE。
- **依赖良好初始化**：若 warm-up 阶段未能提供合理的基函数和系数初值，LM 步骤可能失效。
- **超参数敏感**：Fourier Feature 的带宽 $ \sigma $、阻尼参数 $ \gamma $ 等需根据具体问题调整。

### 未来工作方向

1. **自适应带宽选择**：开发自动调节 $ \sigma $ 的策略，以构建更紧凑高效的基函数。
2. **结合自适应采样**：集成 residual-based adaptive sampling 方法，减少 collocation points 数量，从而降低 LM 子问题的计算负担。
3. **扩展至 Operator Learning**：将 FALM 框架推广到 DeepONet 或 Fourier Neural Operator 等算子学习范式中。
4. **混合架构探索**：结合 Transformer、Graph Neural Networks 等先进架构，进一步提升表达能力。

---

> 📌 **总结**：  
> FALM-PINN 是一种理论严谨、性能卓越的 PINN 新框架。它通过**解耦优化 + Fourier-enhanced basis + LM 求解器**，成功解决了 PINN 在高频率与强非线性 PDE 上的瓶颈问题，在多个基准上实现了**两个数量级以上的误差降低**。尽管存在计算成本较高的问题，但其设计理念为下一代物理信息神经网络的发展提供了重要方向。

</details>

---

### 5. [RepoOMP: Repository-Aware Hotspot OpenMP Parallelization via Dependency-Aware Context Reduction](https://arxiv.org/abs/2608.05855)

**Authors**: Yongjie Qian, Ke Gao, Zhibin Zhang, Shaohui Peng, Ling Li  
**Category**: cs.DC  
**Published**: 2026-08-07  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.05855v1  

#### Abstract
OpenMP parallelization of hotspots in mature repositories remains difficult because loop safety and optimization payoff often depend on non-local evidence. Rule-based tools under-parallelize when legality is not locally provable, while agent-based approaches become unstable when retrieval misses dec...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：RepoOMP: Repository-Aware Hotspot OpenMP Parallelization via Dependency-Aware Context Reduction

## 1. 论文的主要贡献和创新点

### 解决的问题
在大型成熟代码仓库中，对性能热点（hotspot）进行 OpenMP 并行化仍然非常困难。主要原因在于：
- **循环安全性** 和 **优化收益** 往往依赖于非局部证据（如跨文件的共享状态、间接调用、辅助函数副作用等）。
- 基于规则的工具（rule-based tools）在无法静态证明合法性时会保守地拒绝并行化，导致**欠并行化**（under-parallelize）。
- 基于大模型的代理方法（agent-based approaches）在检索缺失关键依赖或包含无关代码时变得不稳定，导致**过并行化**（over-parallelize），引入数据竞争（data race）。

### 提出的新方法和新思路
提出 **RepoOMP**，一个混合框架，通过“依赖感知的上下文压缩”实现仓库感知的热点 OpenMP 并行化。其核心创新包括：

- **Multi-granularity Attributes Performance graph (MAP)**：构建一个仓库级的有向属性图，整合多粒度信息（仓库、文件、函数），显式建模性能热点、调用关系、数据依赖和共享状态访问摘要。
- **Rule-Agent Router**：基于 MAP 中传播的风险信号（如共享写入、I/O、间接访问），将候选热点路由到不同的处理路径：
  - **高置信度**：由确定性规则引擎处理。
  - **中等置信度**：交由 LLM 代理处理。
  - **低置信度**：保守处理，跳过转换。
- **Structured Transformation Context (STC)**：为 LLM 代理构造一个结构化的上下文，仅包含目标代码片段、可见的共享状态定义、符号定义以及关键调用链节点的摘要描述，避免了“信息淹没”和“信息缺失”的双重风险。

### 相比现有方法的优势
- **精准性**：通过 MAP 显式恢复跨文件依赖证据，解决了规则工具因局部不可证而保守拒绝的问题。
- **稳定性**：通过 STC 向 LLM 提供精炼的、任务相关的上下文，避免了通用检索导致的不安全并行化和推理退化。
- **效率**：显著降低了 LLM 代理侧的 token 成本（减少 47-68%），同时提高了验证成功率。
- **系统性**：提供了一个从证据恢复、选择性路由到可执行验证的完整工作流。

---

## 2. 核心实验方法和设置

### 使用的数据集
评估基于 **951 个**经过性能分析的热点，涵盖以下工作负载：
- **微基准套件**：NPB (8 kernels), BOTS (8 kernels)
- **真实世界应用**：
  - **FFmpeg** (视频处理，3 kernels)
  - **NCNN** (神经网络推理框架，3 kernels)
  - **GROMACS** (分子动力学模拟，3 kernels)

### 实验设置和评估指标
- **平台**：双路 Intel Xeon Gold 6430 和单路 AMD EPYC 9654。
- **编译器**：Clang 22.0.0，`-O3`，OpenMP 4.5。
- **评估流程**（三阶段验证）：
  1. **编译检查**（Compilation）：确保生成的代码可编译。
  2. **工作负载特定可执行检查**（Executable Checks）：与黄金基准比较输出（功能/数值检查）。
  3. **性能检查**（Performance）：测量是否带来正向加速。
- **接受标准**：只有同时通过以上三个检查的转换才被视为“接受”（accepted）。
- **性能指标**：
  - **Speedup**：相对于 `-O3` 串行版本的加速比。
  - **Token Usage**：LLM 侧使用的 token 数量（千为单位）。
  - **Acceptance Rate**：被接受的热点比例。

### 基线方法对比
- **传统自动并行化工具**：
  - `AutoPar`：基于源到源转换的规则工具。
  - `Polly`：基于多面体模型的 LLVM 优化器。
- **LLM 代理基线**：
  - `Claude Code`：作为匹配的、无结构的仓库级代理基线（未使用 MAP/STC 路由）。
- **人类专家**：引用 NPB 和 BOTS 的官方人工并行化结果作为上限参考。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- 在 **372 个**通过验证的热点中，包含 **330 个**来自真实世界仓库的热点。
- **平均加速比**：
  - NPB：**8.23×**
  - BOTS：**8.96×**
  - 9 个详细的真实世界内核（matched-backbone 分析）：**5.25×**
- **中位数加速比**：在所有 330 个被接受的真实世界热点中，中位数为 **2.25×**，四分位距（IQR）为 1.45–4.80×。

### 与基线方法的对比结果
- **vs. 传统工具 (AutoPar/Polly)**：
  - AutoPar 和 Polly 在此协议下表现保守，例如在 NPB 上分别仅获得 1.53× 和 1.37× 的平均加速比，远低于 RepoOMP 的 8.23×。
  - 传统工具缺乏从命令行工作负载自动发现热点和转换站点的能力。
- **vs. LLM 代理基线 (Claude Code)**：
  - RepoOMP 将平均加速比从 Claude Code 基线的 **4.17×** 提升至 **4.94–5.46×**。
  - RepoOMP **完全消除了**基线中存在的编译失败（BF）和错误答案（WA）。
  - RepoOMP 将 LLM 代理侧的 token 成本 **降低了 47–68%**（从 111.1k 降至 50.6k–66.4k）。
  - RepoOMP 生成的代码在 ThreadSanitizer 动态竞态检测中产生的警告更少（所有 9 个案例均减少）。

### 消融实验结果
消融研究（Ablation Study）验证了各组件的重要性：
- **No-Router / Agent-Only**：移除路由器后，虽然仍能利用 LLM，但 token 成本和端到端时间大幅增加，且失败率上升。
- **No-STC / Flat-Ctx**：移除 STC 或弱化 MAP 会导致：
  - 接受的热点数量**急剧下降**（例如在 FFmpeg 上从 221 降至 115）。
  - token 成本**显著增加**（例如在 FFmpeg 上从 58.7k 升至 245.3k）。
- **Rule-Only**：仅使用规则引擎，覆盖范围极小（例如在 FFmpeg 上仅接受 33 个热点），证明了 LLM 代理对于处理复杂仓库场景的必要性。

---

## 4. 关键结论和发现

### 主要发现
1. **核心瓶颈是证据恢复**：仓库级 OpenMP 并行化的根本挑战不是生成能力，而是如何在生成前恢复决定转换安全性和收益性的非局部依赖证据。
2. **混合策略优于单一范式**：RepoOMP 通过结合程序分析（MAP）和 LLM（Agent）的优势，实现了比纯规则或纯代理方法更好的效果。
3. **结构化上下文至关重要**：直接向 LLM 提供大量原始代码（unstructured context）是低效且危险的。**STC** 通过有损压缩保留关键事实，是提高效率和稳定性的关键。
4. **路由机制有效控制成本**：Rule-Agent Router 成功地将简单案例分流给低成本规则引擎，只让真正需要语义推理的案例进入高成本的 LLM 流程。

### 方法的局限性
- **预处理开销大**：构建 MAP 需要完整的静态分析和性能剖析，前期成本较高。
- **难以处理复杂动态行为**：对于重度指针别名（heavy pointer aliasing）、复杂的动态内存管理或 I/O 密集型路径，依赖摘要可能过于保守。
- **重构深度有限**：当前方法主要针对 OpenMP 指令插入，对于需要大规模代码重构的并行化模式支持不足。
- **假设编译成功**：框架假设仓库在优化前可以成功编译，对构建配置不稳定的项目支持有限。

### 未来工作方向
- 改进对**指针别名**和**动态内存行为**的依赖分析精度。
- 扩展支持更深层次的**代码重构**，以处理更复杂的并行化需求。
- 将框架扩展到其他优化目标，如 **MPI** 或 **CUDA**，通过改变下游的转换策略。
- 探索更高效的 MAP 构建和增量更新技术，降低预处理成本。

</details>

---

### 6. [Kastor: An efficient fine-tuning strategy for generative emulation of PDE simulations](https://arxiv.org/abs/2608.06107)

**Authors**: Guillaume Couairon, Alexis Jacq, Yu-Han Wu, Renu Singh, Yana Hasson, Quentin Berthet, Romuald Elie  
**Category**: cs.LG  
**Published**: 2026-08-07  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.06107v1  

#### Abstract
Machine learning offers a promising avenue to accelerate physical simulations by replacing computationally expensive traditional Partial Differential Equation (PDE) solvers with fast, differentiable surrogate models. However, standard auto-regressive ML emulators often suffer from error accumulation...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Kastor: An Efficient Fine-Tuning Strategy for Generative Emulation of PDE Simulations

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

传统的 Partial Differential Equation (PDE) 数值求解器（如 Finite Difference 或 Finite Element Methods）计算成本高昂，尤其在需要长期模拟或多次迭代的场景下效率低下。虽然基于 Machine Learning 的 PDE 代理模型（surrogate models）可以加速模拟，但它们面临以下挑战：

- **误差累积**：标准的 auto-regressive 推理方式在长时程预测中容易因误差积累导致发散。
- **生成能力不足**：大多数模型是确定性的，难以捕捉物理系统的随机性或不确定性。
- **物理一致性差**：生成的场可能在空间梯度等衍生量上不准确，影响物理保真度。

### **提出了什么新方法或新思路**

本文提出 **Kastor**，一种高效的微调策略，用于将预训练的物理基础模型（physics foundation model）转化为高性能的生成式 PDE 代理模型。其核心创新包括：

#### ✅ **两阶段推理方案（Two-stage Inference Scheme）**
- **第一阶段**：使用大时间步长（large-stride causal auto-regressive model）进行粗粒度状态预测，减少 auto-regressive 步数，从而降低误差累积。
- **第二阶段**：通过一个轻量级的非因果时间超分辨率网络（non-causal temporal super-resolution network）填充缺失的时间步，恢复完整轨迹。
- 该设计显著提升了长时程预测的稳定性和准确性，同时保持了较低的计算开销。

#### ✅ **均值预测正则化（Mean Prediction Regularization, MPR）**
- 在生成模型训练中引入额外约束：当噪声输入为零时（null noise conditioning），模型应预测输出分布的均值。
- 这使得生成模型既能产生多样化的样本，又能保证对确定性动态模式的高度忠实。
- MPR 被证明对 **Functional Generative Networks (FGN)** 和 **diffusion-based models** 都有效，尤其显著提升了 FGN 的性能。

#### ✅ **梯度差异损失（Gradient Difference Loss, GDL）**
- 引入 GDL 来匹配预测场与真实场之间的空间梯度，而不仅仅是原始场本身。
- 显著提高了生成场的物理保真度，特别是在功率谱密度（power spectrum density）方面表现更优。

### **相比现有方法的优势**

| 方面 | Kastor 的优势 |
|------|----------------|
| **精度** | 显著降低预测误差（平均减少 42.9% fCRPS） |
| **稳定性** | 减少长时程误差累积，提升 rollout 稳定性 |
| **生成质量** | 更好的分布校准（calibration）和光谱一致性 |
| **效率** | 推理速度更快，计算成本更低（约降低 37.5%） |

---

## 2. 核心实验方法和设置

### **使用的数据集**

所有实验均基于 **The Well** 基准数据集中的 **10 个 2D 物理模拟数据集**，涵盖多个领域：

- **Acoustic Scattering**（声波散射）
- **Active Matter**（活性物质）
- **Euler Multi-quadrants**（欧拉方程多象限流动）
- **Gray-Scott Reaction-Diffusion**（格雷-斯科特反应扩散）
- **Planetary Shallow Water Equations (PlanetSWE)**（行星浅水方程）
- **Rayleigh-Bénard Convection**（瑞利-贝纳德对流）

> 注：排除了 3 个不适用的数据集（如时间依赖可解析、轨迹长度不一致等）。

### **实验设置**

- **骨干模型**：基于 **Walrus**（当前最强的物理基础模型）进行微调。
- **输入格式**：输入为历史状态序列 $ x_{t-k:t} $，输出为下一时刻的状态差分 $ d_{t+Δt} $。
- **推理方式**：
  - 主要采用时间步长 $ T=4 $ 的 strided rollout。
  - 对于 Active Matter 和 Rayleigh-Bénard 数据集，保留 $ T=1 $ 因其采样稀疏。
- **生成机制**：
  - 使用 **FGN** 框架，结合 **AdaLN** 注入噪声，并利用 **patch jittering** 作为额外随机源。

### **评估指标**

| 指标 | 描述 |
|------|------|
| **fCRPS (fair Continuous Ranked Probability Score)** | 衡量预测分布与真实分布边缘的一致性，越低越好 |
| **EnsembleMeanRMSE** | 预测均值与真实值之间的 RMSE |
| **VRMSE (variance-normalized RMSE)** | 归一化后的 RMSE，便于跨变量比较 |
| **Spread-Skill Ratio (SpSkR)** | 衡量集合离散度与误差是否匹配，理想值为 1 |
| **Log Spectral Distance (LSD)** | 衡量预测场与真实场的功率谱密度差异，越小越好 |
| **Skill Score** | 相对于基线的相对改进：$ SS = 1 - \frac{M_{\text{model}}}{M_{\text{reference}}} $，越高越好 |

### **基线方法对比**

- **MAE-trained (Walrus baseline)**：使用 MAE 损失训练，辅以 patch jittering 作为集成手段。
- **Diffusion baseline**：将 Walrus 改造为条件去噪器，建立生成式基线。
- **U-Cast**：已有概率化微调方法（CRPS + dropout）。
- **Residual modeling**：残差建模方法作为替代生成策略。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

- **平均预测误差下降**：
  - 相比 Walrus 微调基线，Kastor 在 **fCRPS** 上实现了 **42.9% 的平均误差降低**（见 Figure 1）。
- **VRMSE 表现**：
  - 在 10 个数据集中，有 **8 个优于 Walrus** 的 VRMSE 表现。
- **光谱一致性提升**：
  - GDL 显著改善了 **LSD** 指标，说明生成场的空间频率特性更接近真实物理过程。

### **与基线方法的对比结果**

| 方法 | fCRPS 性能 | 优势说明 |
|------|-----------|---------|
| **Kastor (FGN + MPR + GDL)** | ⭐ 最佳 | 综合性能最优，尤其在长时程预测中 |
| **Diffusion baseline** | 中等偏上 | 生成质量尚可，但推理慢且需多步去噪 |
| **MAE-trained** | 较差 | 确定性模型，无法表达不确定性 |
| **U-Cast** | 不如 Kastor | 即使使用 CRPS，缺乏 MPR 导致稳定性差 |

> 如 Figure 7 所示，**没有 MPR 的 FGN 性能甚至不如 diffusion 或 MAE 模型**，表明 MPR 是关键组件。

### **消融实验结果**

#### 🔹 **MPR 的影响**
- 移除 MPR 后，fCRPS 技能分数大幅下降（平均降低 >15%）。
- 定性结果显示，无 MPR 的生成样本出现明显边界伪影和结构失真。

#### 🔹 **GDL 的影响**
- 加入 GDL 后，LSD 显著改善（光谱误差减少），尤其在具有锐利边界的任务（如 Acoustic Scattering Maze）中效果最明显。
- fCRPS 也进一步提升，说明梯度监督有助于整体预测精度。

#### 🔹 **两阶段推理 vs 标准 AR**
- 使用 $ T=4 $ 的 strided rollout 比 $ T=1 $ 在多数数据集上更优（尤其是 Gray-Scott 和 Euler Quadrants）。
- 结合 upsampling 模块后，在保持高分辨率的同时节省了约 **37.5% 的推理成本**。

#### 🔹 **其他因素**
- **Patch jittering** 是有效的随机性来源，移除后性能显著下降。
- **Muon optimizer** 比 Adam 更适合微调，带来稳定增益。
- **Global normalization** 比 sample-wise normalization 更适合长时程 rollout。

---

## 4. 关键结论和发现

### **主要发现**

1. **从确定性模型到生成式模型的转化是可行且高效的**：
   - 只需合理设计训练目标（如 MPR），即可将强大的预训练物理模型转化为高质量的概率预测器。

2. **MPR 是生成式 PDE 模拟的关键正则项**：
   - 它强制模型学习“确定性核心”，从而提高生成样本的质量和稳定性。
   - 尤其适用于 FGN 架构，使其超越更复杂的 diffusion 模型。

3. **两阶段推理兼顾效率与精度**：
   - 大步长 rollout 减少误差传播，超分辨率模块恢复细节，是一种实用且高效的设计。

4. **物理感知损失函数至关重要**：
   - 单纯优化像素级误差（如 MAE）不足以保证物理一致性；加入 GDL 可显著提升光谱保真度。

### **方法的局限性**

- **校准仍不完美**：尽管 MPR 提升了 SpSkR，但部分任务中集合仍存在轻微过分散或欠分散现象。
- **初始上下文依赖固定长度**：目前仍需固定数量的历史帧作为输入，尚未实现从单帧逐步扩展上下文。
- **极端高频伪影风险**：diffusion 模型在过多去噪步数时可能出现高频噪声，需配合 GDL 缓解。

### **未来工作方向**

- 探索 **自适应时间步长策略**：例如前期用 $ T=1 $，后期切换至 $ T=4 $。
- 实现 **从单一初态开始 rollout** 并动态增长上下文窗口。
- 将 Kastor 扩展至 **3D 和更高维 PDE 系统**。
- 探索 **test-time adaptation** 或 **feedback control** 机制以进一步增强稳定性。

---

> ✅ **总结一句话**：  
> Kastor 通过 **两阶段推理 + MPR 正则化 + GDL 损失** 的组合，成功地将预训练物理模型高效转化为高精度、高稳定性的生成式 PDE 代理模型，在多个维度上全面超越现有方法。

</details>

---

### 7. [Operating Multi-Node Full Fine-Tuning on NVIDIA B300: A Field Report on Telemetry-Based Triage, Negative Results, and Operational Hardening](https://arxiv.org/abs/2608.05944)

**Authors**: Seon Ho Kim, Ui Jeong Jeon, Su Hyeon Kim, Min Tae Hwang  
**Category**: cs.DC  
**Published**: 2026-08-07  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.05944v1  

#### Abstract
We report operational experience full-fine-tuning a 32.76B-parameter dense model (Qwen3-32B) on 16 x NVIDIA B300 (two nodes, FSDP / ZeRO-3) -- among the first published field accounts on this accelerator. We claim no new algorithm. The individual mechanisms we use are established practice; our contr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Operating Multi-Node Full Fine-Tuning on NVIDIA B300*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文聚焦于在新型硬件 **NVIDIA B300** 上进行大规模 **多节点全参数微调（full fine-tuning）** 的实际运维挑战。尽管已有成熟的并行训练框架（如 FSDP/ZeRO-3），但在真实生产环境中仍面临以下关键问题：

- **GPU utilization% 的误导性**：传统监控指标（如 `nvidia-smi` 显示的 GPU 利用率）无法准确反映训练进度，在 NCCL 死锁等场景下仍显示为 100%，导致难以快速诊断故障。
- **静默失败成本高昂**：一次 epoch 结束时的死锁可能导致数小时的 GPU 小时浪费（~130 GPU·h）。
- **常见优化直觉失效**：例如本地缓存预分词数据是否真能提升吞吐？梯度检查点是否总是有益？
- **数据依赖型任务中的隐性不均衡风险**：由于样本长度差异导致各 rank 处理的 token 块数量不同，可能引发集体通信死锁。

### 提出了什么新方法或新思路
作者并未提出新的算法或模型架构，而是从**操作实践（operational discipline）** 角度出发，提出了四项面向实践者的可复用成果（practitioner artifacts）：

1. **基于板级功耗（board power）的 triage 表**  
   - 使用 **GPU power draw** 而非 utilization 来判断运行阶段（计算、通信、饥饿、死锁、空闲）。
   - 给出 B300 硬件校准的具体瓦数区间，并结合 per-rank 利用率分布形状进行精确分类。

2. **诚实的负向结果（honest negative results）**  
   - 揭示某些“常识”优化无效：当数据集可完全载入 page cache 时，**本地 NVMe 缓存不会带来吞吐增益**。
   - 修正早期误判：“吞吐崩溃”实为 CPU/NFS 争用所致，而非存储介质瓶颈。

3. **B300 上的强扩展性基准数据（calibrated strong-scaling reference）**  
   - 提供 4/8/16 GPU 下完整的 epoch 性能与 GPU·hour 数据，作为后续研究的参考基线。

4. **针对 epoch-end 死锁的操作加固方案（operational hardening）**  
   - 提出 `evenfix`：在训练前通过 `all_reduce(MIN)` 对齐所有 rank 的最大步数，防止因 token packing 不均导致的 collective 错配。
   - 设计 **2.7 秒启动前门控（preflight gate）** 和 **外部观察者（external watcher）**，将潜在的多小时静默失败转化为即时拒绝。

### 相比现有方法的优势
| 方面 | 传统做法 | 本文改进 |
|------|----------|-----------|
| 故障诊断 | 依赖 utilization%，需 Profiler 分析 | 仅凭 `nvidia-smi` 功耗 + 形状即可秒级定位 |
| 数据管道优化 | 默认使用本地缓存加速 I/O | 验证后指出 page cache 已足够，避免过度工程 |
| 死锁预防 | 事后调试、重启 | 启动前验证不变量（invariant），结构性杜绝 |
| 成本控制 | 单次失败损失 ~130 GPU·h | 2.7s 检查避免灾难性浪费 |

> ✅ **核心思想转变**：  
> **“utilization lies, power tells the truth”**  
> **“a passing smoke test is not evidence of a safe full run”**

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Nemotron-Personas-Korea**
  - 包含 1,000,000 条记录
  - 每 epoch 约 **1.01B tokens**
  - 经过一次性分词并打包成固定长度 block 存储

### 实验设置
| 组件 | 配置 |
|------|------|
| **Hardware** | 16 × NVIDIA B300 SXM6（2 节点 × 8 GPU），每卡 275GB HBM，TGP 1100W |
| **Interconnect** | InfiniBand NDR + GPUDirect RDMA，节点内 NVLink |
| **Software** | NGC 容器（PyTorch 2.12, Transformers 5.12, native FSDP） |
| **Model** | Qwen3-32B（32.76B 参数，dense） |
| **Parallelism** | FSDP FULL_SHARD（等价于 ZeRO-3），world size = 16 |
| **Memory Footprint** | ~82 GB/GPU（激活 + 分片模型状态），占 B300 内存的 ~30% |
| **Sequence Length** | 2048 |
| **Micro-batch Size** | 1 |
| **Gradient Accumulation** | 8 → Global Batch = 128 seq / 262,144 tokens/step |
| **Steps per Epoch** | ~3,854 |
| **Checkpointing** | 每 500 步保存一次，epoch 结束时额外保存 |

### 评估指标
| 指标 | 描述 |
|------|------|
| **Throughput (tok/s)** | 每秒处理的 token 数量，主性能指标 |
| **GPU Utilization (%)** | `nvidia-smi` 输出，用于对比其误导性 |
| **Board Power Draw (W)** | 每 GPU 实际功耗，作为核心 triage 依据 |
| **Strong-Scaling Efficiency** | 相对于 4-GPU 基线的加速比 |
| **GPU·hour per epoch** | 单 epoch 所需总计算资源 |
| **Failure Detection Time** | 从发生 hang 到被识别的时间 |
| **Cost of Silent Failure** | 单次死锁造成的 GPU·h 浪费估算 |

### 基线方法对比
| 场景 | 对比方式 |
|------|---------|
| 数据路径优化 | A/B 测试：<br>A: NFS 实时读取 + 分词（冷/热缓存）<br>B: 预分词 + 本地 NVMe 缓存 |
| 梯度检查点 | 开启 vs 关闭 gradient checkpointing |
| 死锁规避 | 有无 `evenfix` 的完整 epoch 运行成功率 |
| 扩展性 | 4/8/16 GPU 强扩展效率 vs 理想线性 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 🔹 吞吐量与扩展性（B300）
| GPUs | Nodes | tok/s | tok/s/GPU | Epoch Time (h) | GPU·h/epoch | Strong-Scaling Eff. |
|------|-------|--------|------------|----------------|--------------|------------------------|
| 4    | 0.5   | 13.4k  | 3,350      | 20.9           | 83.7         | 100%                   |
| 8    | 1     | 26.7k  | 3,338      | 10.5           | 84.1         | 100%                   |
| 16   | 2     | **53.0k** | **3,313**  | **5.3**        | **84.7**     | **99%**                |

- ✅ **近似线性扩展**：16 GPU 达到 99% 效率，通信开销极小（隐藏在计算中）
- ✅ **每 GPU 吞吐稳定**：3,313–3,350 tok/s/GPU，波动 <1%
- ✅ **总工作量守恒**：~84 GPU·h/epoch，表明无显著额外开销

#### 🔹 数据路径 A/B 测试结果（16 GPU）
| 路径 | 条件 | Median tok/s | 相对性能 |
|------|------|---------------|-----------|
| NFS + 分词 | 冷缓存（drop caches 后） | 52.8k | 1.00× |
| NFS + 分词 | 热缓存（page cache 命中） | 53.0k | 1.00× |
| 本地预缓存 | packed i32 文件 | **53.0k** | **1.00×** |

- ❌ **本地缓存未提速**：因数据仅 ~4GB，远小于单节点 4TB RAM，始终驻留 page cache
- ⚠️ 仅首次访问慢（33.9k tok/s），JIT 编译 + 缓存填充影响，10 步内恢复

#### 🔹 梯度检查点影响
| 设置 | Throughput | 备注 |
|------|-----------|------|
| gradient checkpointing **off** | **53k tok/s** | 推荐（内存充足） |
| gradient checkpointing **on** | 33k tok/s | 损失 ~1.6× 吞吐 |

> 💡 结论：**“Always enable gradient checkpointing” 是内存受限下的建议；若内存充裕，则纯属浪费**

#### 🔹 死锁事件分析
| 指标 | 数值 |
|------|------|
| 发生位置 | Step 3,850 / 3,854（epoch-end checkpoint） |
| 症状 | 所有 GPU utilization=100%，但 throughput=0 |
| 实际功耗 | ~190W（一卡 0%，其余 100%）→ 死锁特征 |
| 根因 | Rank 3 先耗尽数据退出训练循环，进入 `dist.barrier()`，而其他 rank 仍在执行 backward reduce_scatter |
| Block Count 差异 | min=30,835（rank 3），max=30,888（rank 11），差 53 块（0.17%） |
| `evenfix` 截断总量 | ≤53 blocks/rank，≈ **0.085% tokens 被丢弃**，可忽略 |

#### 🔹 操作加固效果
| 组件 | 效果 |
|------|------|
| `evenfix`（all_reduce(MIN)） | 结构性消除死锁可能 |
| **Preflight Gate**（6项检查） | 耗时 **2.71 秒**，阻止 unsafe launch |
| External Watcher | 可检测 hang/crash/checkpoint stall 并报警 |
| 单次失败成本（pre-fix） | **~130 GPU·h**（≈1.5 个完整 epoch） |

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Watch Power, Not Utilization**  
   GPU utilization % 在 NCCL hang、I/O wait、checkpoint 等场景下完全不可信。**板级功耗 + per-rank 利用率分布** 是更可靠的运行状态信号。

2. ✅ **Page Cache 已足够，无需本地 SSD 缓存**  
   当数据集 << RAM 时，NFS + page cache 可达与本地 NVMe 相同吞吐。**优化应优先考虑 determinism 与 robustness，而非速度**。

3. ✅ **所谓“存储瓶颈”实为 CPU/NFS 争用**  
   早前观察到的吞吐下降是因单线程分词 + 多租户 NFS 负载造成，**并非存储介质本身限制**。

4. ✅ **Gradient Checkpointing 在内存富裕时不适用**  
   开启反而降低 ~1.6× 吞吐。应在有明确 memory pressure 时才启用。

5. ✅ **Even Miniature Imbalance Can Kill**  
   仅 **0.17% 的 per-rank block count 差异** 就足以引发 epoch-end 死锁，且 smoke test 无法捕获（因其不触及边界条件）。

6. ✅ **Prevention Beats Debugging**  
   一个 **2.7 秒的启动前门控** 可防止价值 ~130 GPU·h 的失败，经济性极高。

7. ✅ **FSDP 不支持 Join Context Manager**  
   PyTorch 的 `Join` 机制目前仅适用于 DDP 和 ZeRO，**FSDP 用户必须手动对齐 step count**。

8. ✅ **SHARP 收益有限于当前规模**  
   因通信已完全隐藏在计算中（99% scaling efficiency），switch-level aggregation（如 SHARP）最多带来 <1% 提升。

---

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **硬件特定性** | 功耗阈值（如 940W、575W）为 B300 特有，不可直接迁移至其他 GPU |
| **规模限制** | 实验仅到 16 GPU / 2 节点，不能外推至千卡级别可靠性行为 |
| **模型单一** | 仅测试 Qwen3-32B，结论对 MoE 或更大模型未必成立 |
| **数据假设强** | “本地缓存无收益” 结论依赖于 dataset ≪ RAM，超大 dataset 场景不适用 |
| **成本为估算** | 失败代价基于日志重建，非严格对照实验 |

---

### 未来工作方向
1. **推广 power-based triage 至更多硬件平台**  
   建立跨 GPU 架构的标准化功耗诊断谱系。

2. **开发自动化 invariant checker 框架**  
   将 `preflight gate` 抽象为通用工具，支持多种训练任务。

3. **探索 packing-time balance 算法**  
   如 SlimPack、Hierarchical Balance Packing，在源头解决 block 不均问题。

4. **增强 FSDP 的 Join 支持**  
   推动 PyTorch 社区实现 FSDP-compatible Joinable 接口。

5. **构建 telemetry-driven auto-recovery pipeline**  
   结合 external watcher 与 checkpoint 自动回滚，实现无人值守长周期训练。

6. **研究更大规模下的 SHARP 价值**  
   在 >64 节点场景验证 in-network aggregation 是否突破通信墙。

---

> 📌 **最终启示录**：  
> 在分布式深度学习中，**最危险的不是错误，而是你以为一切正常**。  
> 利用硬件 telemetry、坚持前置验证、拥抱负向结果——这是高可靠训练系统的真正基石。

</details>

---

### 8. [Disentangling 3D Modeling from Spatial Reasoning](https://arxiv.org/abs/2608.05242)

**Authors**: Haoze Sun, Jiequan Cui, Qingshan Xu, Richang Hong  
**Category**: cs.LG  
**Published**: 2026-08-07  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.05242v1  

#### Abstract
In this work, we explore an alternative paradigm for spatial reasoning by explicitly disentangling 3D perception from reasoning, rather than jointly acquiring implicit 3D perception and reasoning through large-scale training. Our key observation is that modern perception models excel at estimating c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Disentangling 3D Modeling from Spatial Reasoning*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

当前主流的 **Multimodal Large Language Models (MLLMs)** 在实现空间推理（spatial reasoning）时，通常采用**端到端联合训练**的方式，在大规模 3D VQA 数据上隐式地学习几何感知与空间关系推理。这种方法存在以下问题：

- **耦合性强**：将 3D 感知与推理能力捆绑在同一个模型中，导致知识难以解释、错误难以诊断。
- **计算成本高**：需要大量 3D 标注数据和强大的算力进行训练。
- **可扩展性差**：改进感知模块需重新训练整个系统。

### 🚀 提出的新方法与新思路

作者提出 **Disentangled Spatial Reasoner (DiSR)**，其核心思想是**显式解耦 3D 感知与空间推理**，构建一个两阶段框架：

1. **Structured 3D Evidence Construction**  
   利用现成的专家级感知模型（off-the-shelf expert perception models）从图像中提取结构化的 3D 几何证据，包括：
   - 对象中心（center）
   - 尺寸（extent）
   - 方向（front/left axes）
   - 相机姿态与度量深度（metric depth）

2. **Reasoning over Structured 3D Evidence**  
   冻结所有感知模型，仅对 LLM 使用 **LoRA** 进行微调，使其基于上述结构化文本形式的 3D 证据进行符号化、组合式推理，输出答案。

> 💡 关键洞察：现代感知模型擅长连续几何估计，而 LLM 擅长符号推理 —— 应让各自专精，而非强求统一模型掌握两者。

### ⭐ 相比现有方法的优势

| 优势维度 | 具体体现 |
|--------|--------|
| **高效性** | 仅需 0.33M 训练样本、单张 RTX 4080 SUPER 训练 59 小时，远低于 HiSpatial（2B 数据 + 32×H100） |
| **可解释性** | 中间生成的结构化 3D 证据可被人工检查，支持错误归因（如区分是感知错还是推理错） |
| **模块化** | 可独立升级感知模型或 LLM，无需重新训练整个系统 |
| **轻量化** | 仅微调 LoRA 参数，保留原始 MLLM 的通用视觉理解能力 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 数据集 | 任务描述 |
|------|-------|
| **3DSRBench** (Ma et al., 2025a) | 细粒度 3D 空间推理评测，涵盖高度、位置、朝向、多对象关系等 |
| **SPAR-Bench** (Zhang et al., 2026) | 包含深度估计（Depth-OC/OO）、距离估计（Dist-OC/OO）、关系选择（Relational Selection） |
| **CV-Bench-3D** (Tong et al., 2024) | 聚焦超越二维平面线索的定性深度与距离推理 |
| **General Benchmarks** | MMBench、GQA、POPE、SEED、RealWorldQA —— 验证通用视觉推理能力是否退化 |

### 📊 实验设置与评估指标

| 设置项 | 描述 |
|------|-----|
| **主干模型** | Qwen3-VL-8B-Instruct（用于 object grounding + LLM reasoning） |
| **感知组件** | SAM（mask）、Metric3D（depth）、WildCamera + PerspectiveFields（camera pose）、OrientAnything（orientation） |
| **微调方式** | LoRA（rank=128, α=256），冻结所有感知模型 |
| **训练数据量** | 0.33M 样本，来源于 Open Images 自动生成的空间问答对 |
| **评估指标** |  
- 数值任务：Mean Relative Accuracy (**MRA**)  
- 分类任务：Accuracy (**Acc.**)  

### 🔁 基线方法对比

分为三类：

1. **开源通用 MLLMs**  
   - LLaVA, Cambrian, Qwen-VL 系列

2. **闭源商业模型**  
   - GPT-4o, Claude-3.5-Sonnet, Gemini, QwenVLMax

3. **空间专项模型**  
   - SpatialRGPT, SpatialReasoner, HiSpatial, SpaceLLaVA, SpatialPIN, SpaceTools 等

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ✅ 在 3DSRBench 上的表现（Table 1）

| 方法 | Overall Acc. |
|------|-------------|
| **DiSR-8B-LoRA** | **67.62%** |
| HiSpatial-3B | 63.85% |
| SpatialReasoner | 60.30% |
| Qwen3-VL-8B-Instruct | 53.70% |

👉 **提升 +3.77%**，达到 SOTA，尤其在 **Multi-Object Relations** 上显著领先（+11.5% 以上）。

#### ✅ 在 SPAR-Bench 上的表现（Table 2）

| 方法 | Metric Estimation (MRA) | Relational Selection (Acc.) | **Avg.** |
|------|--------------------------|-------------------------------|---------|
| **DiSR-8B-LoRA** | **49.37%** | 40.25% | **46.33%** |
| HiSpatial-3B | 29.10% | 70.12% | 42.78% |
| Qwen3-VL-8B-Instruct | 33.55% | 66.81% | 44.63% |

👉 **总体高出 1.70%**，在**度量估计任务上大幅领先**（+20.27% vs HiSpatial），但在关系选择任务上较弱（因训练数据未覆盖该分布）。

#### ✅ 在 CV-Bench-3D 上的表现（Table 3）

| 方法 | Depth | Distance | Avg. |
|------|-------|----------|------|
| Qwen3-VL-8B-Instruct | 95.50 | 89.83 | 92.67 |
| **DiSR-8B-LoRA** | 92.83 | 91.67 | **92.25** |
| DiSR w/ GT Grounding | 94.17 | 93.33 | 93.75 |

👉 性能与基础模型相当，略低但差距小；引入真实框后可达 93.75%，说明瓶颈在于 object grounding。

### 🔍 消融实验结果

#### （1）错误归因分析（Table 5）

| 设置 | Depth Acc. | Distance Acc. |
|------|------------|---------------|
| DiSR-8B-LoRA | 92.83 | 91.67 |
| + GT Question Parsing | 92.33 | 92.83 |
| + GT Object Grounding | 94.17 | 93.33 |
| + GT 3D Evidence | **100.0** | **100.0** |

👉 表明：**只要提供准确的 3D 证据，LLM 推理几乎无误**，当前误差主要来自感知阶段（尤其是 object grounding 和小物体建模）。

#### （2）跨骨干验证（Table 7）

| Backbone | Overall Acc. on 3DSRBench |
|---------|----------------------------|
| Qwen2.5-VL-7B-Instruct | 65.52% |
| Qwen3-VL-8B-Instruct | **67.62%** |

👉 显示 DiSR 框架具有良好的**可迁移性和扩展性**，更强的 backbone 能直接带来性能增益。

#### （3）object grounding 能力保持（Table 6）

| Model | mIoU | IoU@0.75 |
|-------|------|----------|
| Qwen3-VL-8B-Instruct | 71.05 | 66.39 |
| **DiSR-8B-LoRA** | **73.99** | **68.17** |
| SpatialReasoner | 34.82 | 7.07 |

👉 DiSR 不仅没有损害定位能力，反而略有提升；而部分专用模型（如 SpatialReasoner）虽有强下游表现，却严重退化了 grounding 能力。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **显式解耦优于隐式联合建模**  
   将 3D 感知与空间推理分离，不仅能获得更优性能，还能大幅提升**可解释性、模块化与训练效率**。

2. **LLM 擅长符号推理而非几何学习**  
   当提供准确的结构化 3D 输入时，LLM 可以近乎完美地完成复杂空间推理，无需从图像中“学”几何。

3. **高性能 ≠ 强泛化**  
   如 HiSpatial 在 CV-Bench-3D 表现好，但其 object grounding 失败，说明可能存在“捷径学习”（shortcut learning），而 DiSR 更可靠。

4. **存在明显的 2D Shortcut 问题**  
   - CV-Bench-3D 中存在强烈 2D 线索（如 bottom-position、center proximity），许多模型通过这些线索“作弊”。
   - DiSR 对冲突样本（2D cue 与真实 3D 不一致）表现更好，证明其依赖的是真正的 3D 结构。

5. **小物体是当前感知瓶颈**  
   在 CV-Bench-3D 中，DiSR 的失败案例集中在**小目标区域**，其 3D 重建质量较差（Appendix Table A4）。

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **依赖高质量感知模型输出** | 若感知模型失效（如遮挡、小物体、低分辨率图像），整体性能下降明显 |
| **无法处理非结构化或动态场景** | 当前框架假设静态、刚体对象，难以应对形变、透明物体等复杂情况 |
| **关系选择任务表现弱** | 因训练数据未充分覆盖此类任务，需针对性增强 |
| **仍需预定义对象查询机制** | question parsing 依赖语言模型，可能引入歧义 |

### 🔮 未来工作方向

1. **集成更强的感知模型**  
   引入更高精度的小物体检测与重建技术，缓解当前瓶颈。

2. **支持交互式输入**  
   用户可手动指定感兴趣对象（模拟 w/ GT grounding），进一步提升实用性。

3. **构建面向 DiSR 的专用训练数据集**  
   特别针对关系选择、冲突样本设计数据，提升鲁棒性。

4. **探索闭环控制应用**  
   将 DiSR 部署于机器人导航、操作等实际任务中，验证其在真实环境中的有效性。

5. **拓展至动态与开放世界场景**  
   支持时间序列推理、运动预测等更复杂的时空理解任务。

---

> 📌 **一句话总结**：  
> **DiSR 证明了“专业化分工”在空间智能中的巨大潜力——让感知模型专注建模 3D，让 LLM 专注逻辑推理，通过显式结构化解耦，实现了更高效、可解释且高性能的空间推理新范式。**

</details>

---

### 9. [Hyper-ES: Effective Evolution Strategies for LLM Reasoning via Descent Direction Merging](https://arxiv.org/abs/2608.05541)

**Authors**: Yu Gu, Zhi Zheng, Yunpeng Ba, Xialiang Tong, Mingxuan Yuan, Zhenkun Wang  
**Category**: cs.AI  
**Published**: 2026-08-07  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.05541v1  

#### Abstract
Evolution Strategy (ES) is a promising alternative to gradient-based fine-tuning for resource-constrained Large Language Model (LLM) reasoning. However, directly applying ES to billion-parameter LLMs is highly ineffective. In such high-dimensional parameter spaces, most random perturbations are near...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Hyper-ES: Effective Evolution Strategies for LLM Reasoning via Descent Direction Merging**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
- **Evolution Strategy (ES)** 是一种无需梯度的优化方法，在资源受限场景下对大模型进行微调具有内存效率优势（如 10× 更少的 GPU 内存）。
- 然而，直接将 ES 应用于 **billion-parameter 级别的 LLM 参数空间**时存在严重缺陷：
  - **高维随机游走（High-dimensional random walk）**：大多数随机扰动方向与有效的下降方向近乎正交，导致搜索效率极低。
  - **方向发现失败（Directional discovery failure）**：有限种群无法在超高维空间中有效捕捉到任务相关的更新方向。
  - **参数漂移（Parameter drift）**：无意义的正交更新累积造成模型远离预训练权重，影响稳定性。

### 🚀 提出的新方法：**HYPER-ES**
- **核心思想**：将全参数空间的 ES 搜索转换为一个**低维、结构化的子空间搜索**，该子空间由多个短梯度步生成的“下降方向”张成。
- **两阶段流程**：
  1. **启动阶段（Start-up）**：
     - 将训练数据划分为 $N$ 个子集。
     - 在每个子集上运行少量（如 7 步）的 **GRPO + LoRA** 微调，得到 $N$ 个 LoRA 增量 $\Delta \theta_i$。
     - 这些增量构成一个**任务相关下降方向池（Descent Direction Pool）**。
  2. **ES 搜索阶段**：
     - 固定原始模型和方向池。
     - 使用 **CMA-ES** 在这些方向的组合系数上进行黑盒优化。
     - 具体地，搜索变量 $z$ 控制每层的：
       - **DARE** 的 drop rate（稀疏化）
       - **TIES** 的 mixing weight（加权融合）
     - 最终通过 `DARE-TIES` 合并机制生成最终模型。

### 🔍 相比现有方法的优势
| 方面 | 优势说明 |
|------|----------|
| **效率** | 避免了全参数梯度回传，仅需少量梯度步骤构建方向池，后续为纯前向推理 + CMA-ES，节省 10% 以上的梯度计算量。 |
| **有效性** | 在低维、任务对齐的子空间中搜索，避免了高维随机游走，显著提升样本效率和收敛稳定性。 |
| **性能** | 在数学推理任务上，**一致优于 GRPO+LoRA 和 CMA-ES+LoRA**，平均提升约 1%。 |
| **通用性** | 可推广至代码生成等非数学任务，并保持优势。 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 类型 | 数据集 |
|------|--------|
| **训练数据** |  
| - 数学推理 | `GSM8K-Aug`, `DeepScaler`（含推理步数信息） |
| **测试数据** |  
| - 算术推理 | `GSM8K`, `SVAMP`, `MultiArith`, `GSM-Hard` |
| - 高难度数学 | `MATH-500`, `AMC2023` |
| - 代码生成（额外验证） | `MBPP`, `HumanEval` |

### ⚙️ 实验设置
- **模型基座（Backbone）**：
  - `Qwen2.5-0.5B-Instruct`
  - `Qwen2.5-1.5B-Instruct`
  - `DeepSeek-R1-Distill-Qwen-1.5B`
- **LoRA 设置**：
  - Rank: 32, Scaling: 64 ($\alpha/r = 2.0$)
  - Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `down_proj`, `gate_proj`, `up_proj`
- **方向数量**：
  - Qwen2.5: $N=10$
  - DeepSeek: $N=7$
- **每方向 GRPO 步数**：7 步，batch size 256 → 总梯度样本量远小于完整 GRPO 训练。

### 🎯 评估指标
| 模型 | 指标 |
|------|------|
| Qwen2.5 系列 | **Accuracy (%)**（greedy decoding, $T=0$） |
| DeepSeek-R1-Distill | **Mean@32 (%)**（32 次采样下的平均 Pass@1，降低方差） |
| 代码生成 | **Pass@1** on `MBPP` / `HumanEval` |

### 🆚 基线方法对比
| 基线 | 描述 |
|------|------|
| **Base** | 未经微调的原始模型 |
| **GRPO+LoRA** | 完整数据集上的标准 RL 微调，作为强基线 |
| **CMA-ES+LoRA** | 直接在 LoRA 参数空间应用 CMA-ES，无方向引导 |
| **Average Merge** | 对所有 LoRA 增量取平均合并 |
| **DARE+CMA-ES** | 固定全局超参，仅用 CMA-ES 调整系数（无 layer-wise 自适应） |
| **tinyGRPO**, **CABS**, **WUDI-Merging** | 其他轻量级或模型合并基线（见附录） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Tables 1–2）

#### ✅ 数学推理任务表现（平均 Accuracy / Mean@32）

| 方法 | Qwen2.5-0.5B | Qwen2.5-1.5B | DeepSeek-1.5B |
|------|--------------|--------------|----------------|
| GRPO+LoRA | 56.23% | 73.51% | 70.39% |
| CMA-ES+LoRA | 52.76% | 72.03% | 69.63% |
| **HYPER-ES (Ours)** | **57.13%** (+0.90) | **74.26%** (+0.75) | **70.97%** (+0.58) |

> 💡 **结论**：HYPER-ES 在三种 backbone 上均**稳定超越 GRPO+LoRA**，最大提升达 **1% 绝对增益**。

#### 🔍 在最具挑战性的 MATH-500 上的表现
| 方法 | MATH-500 分数 |
|------|---------------|
| GRPO+LoRA | 75.48% |
| **HYPER-ES** | **76.97%** (+1.49%) |

> 表明其在复杂数学推理任务上有更强泛化能力。

---

### 🔬 消融实验结果（Table 3 & Table 4）

#### Ablation Study on Qwen2.5-0.5B

| 变体 | 平均准确率 | Δ 相比完整版 |
|------|------------|-------------|
| **HYPER-ES (完整)** | 57.13% | +0.00 |
| w/o CMA-ES（仅 grid search） | 55.92% | -1.21 |
| w/o Grouping（随机分组） | 53.62% | -3.51 |

> ✅ **CMA-ES 层级自适应系数至关重要**  
> ✅ **基于难度的数据分组（Grouping）显著影响方向质量**

#### 梯度预算对比（Table 4）

| 方法 | 梯度样本数 | 平均准确率 |
|------|------------|------------|
| GRPO+LoRA | 20,000 | 56.23% |
| HYPER-ES (7步/方向) | 17,920 (**↓10.4%**) | **57.13%** |
| HYPER-ES (4步/方向) | 10,240 (**↓48.8%**) | 55.71% |

> 💡 **用更少的梯度步即可获得更好或相当的结果**，证明了方向构造的高效性。

#### 控制变量：方向质量的重要性
| 方法 | 准确率 |
|------|--------|
| Pure random directions | 47.58%（接近 Base） |
| CMA-ES on random LoRA dirs | 50.97% |
| **HYPER-ES (task-aligned dirs)** | **57.13%** |

> ❗ **关键不是低维搜索本身，而是搜索空间必须是任务对齐的（task-aligned）**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **高维 ES 失败的根本原因是几何错配**：
   - 随机扰动几乎总是与真实下降方向正交（Lemma 1）。
   - 正交更新会累积成参数漂移（Lemma 2），导致不稳定。

2. **HYPER-ES 成功的关键在于“解耦”**：
   - **梯度方法用于发现方向**（cheap & informative）
   - **ES 方法用于选择组合**（efficient & stable）

3. **性能提升主要来自 CMA-ES 阶段**：
   - 即使是弱方向，通过 DARE-TIES + CMA-ES 的 layer-wise 系数优化也能实现超越。
   - 如 Qwen2.5-1.5B 上：`GRPO → Grid-only → CMA-ES` 分别达到 74.22% → 75.21% → 75.51%

4. **高度可并行化，节省端到端时间**：
   - 多个 few-shot GRPO 可并行执行。
   - 实测在 Qwen2.5-0.5B 上总耗时从 4.25h 降至 3.23h（↓24%）。

---

### ⚠️ 方法的局限性
1. **依赖 GRPO 作为方向生成器**：
   - 当前仅验证了 GRPO 的有效性，未探索其他初始化方式（如 SFT 或 OPD）是否同样适用。
2. **静态方向池**：
   - 所有方向在 ES 前固定，缺乏在线动态更新机制。
3. **方向数量与质量平衡**：
   - 过多方向可能导致冗余，过少则限制表达能力，需经验设定 $N$。

---

### 🔮 未来工作方向
1. **扩展方向来源**：
   - 探索使用 SFT、DAPO 或其他 RL 方法生成初始方向。
2. **动态方向生成**：
   - 设计 online 机制，在 ES 过程中迭代改进方向池。
3. **理论分析子空间维度与性能关系**：
   - 研究最优 $N$ 与任务复杂度的关系。
4. **应用于更多下游任务**：
   - 如多模态推理、规划、Agent 决策等。

---

> 🔗 **开源地址**：[https://github.com/kuangrepi/Hyper-ES](https://github.com/kuangrepi/Hyper-ES)  
> 📘 **一句话总结**：**HYPER-ES 将 Evolution Strategy 从“盲目随机探索”转变为“智能融合已有智慧”，实现了高效、稳定且高性能的 LLM 推理能力增强。**

</details>

---

### 10. [iARCS: Iterative Agentic RL for Controllable 3D Scene Generation](https://arxiv.org/abs/2608.06161)

**Authors**: Saugat Adhikari, Ashok Prasad Neupane, Pramish Paudel, Ajad Chhatkuli, Danda Pani Paudel  
**Category**: cs.AI  
**Published**: 2026-08-07  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.06161v1  

#### Abstract
Synthetic 3D scene generation is increasingly used as a data source for computer vision and embodied AI, but existing generators often optimize perceptual realism without reliably satisfying task-critical functional constraints. This mismatch limits the usefulness of synthetic data for downstream tr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**iARCS: Iterative Agentic RL for Controllable 3D Scene Generation**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决了什么问题
现有的 **3D 场景生成模型**（如 ATISS、MiDiffusion）虽然在感知真实性和分布匹配方面表现良好，但在满足**任务关键的功能性约束**（如可达性、可行走性、空间规则）方面存在明显不足。这导致合成数据在用于下游任务（如机器人导航、人机交互）时实用性受限。

具体问题包括：
- 文本引导生成难以精确执行几何约束（如距离、角度、视线畅通）。
- 手工设计奖励函数成本高、泛化差，难以扩展到任意新约束。
- 传统方法缺乏对非可微目标的有效优化机制。

---

### ✅ 提出了什么新方法或新思路
提出 **iARCS**（Iterative Agentic RL for Controllable 3D Scene Generation），一种基于**迭代代理式强化学习**（Agentic Reinforcement Learning）的可控 3D 场景生成框架，其核心思想是：

#### 🌟 两阶段训练策略（Two-stage Training Strategy）
1. **Stage 1: Universal-Reward Pretraining**
   - 使用一组通用奖励（如防碰撞、边界贴合、可达性、密度正则化）对预训练扩散模型进行 RL 微调。
   - 目标：提升基础物理合理性和布局质量，建立高质量场景先验。

2. **Stage 2: Task-Specific Fine-Tuning with Agentic Feedback**
   - 引入 **LLM Agent** 将自然语言指令转化为可执行的奖励程序（`reward function`）。
   - 通过 **Denoising Diffusion Policy Optimization (DDPO)** 优化复合奖励目标。
   - 加入 **Reward Reflection 模块**：定期分析生成结果与奖励统计，动态修正奖励代码或权重，防止“reward hacking”。

#### 🌟 创新机制
- **Agentic Reward Synthesis**：利用 LLM 自动将自然语言需求分解为可量化的几何检查项，并生成 Python 可执行奖励函数。
- **Iterative Refinement Loop**：结合训练反馈持续改进奖励定义，实现自适应优化。
- **Post-training Adaptation 范式**：不从零训练模型，而是对已有扩散先验进行任务适配，高效且实用。

---

### ✅ 相比现有方法的优势
| 维度 | 现有方法局限 | iARCS 改进 |
|------|---------------|-----------|
| **可控性** | 文本条件控制弱，无法保证精确几何约束 | 支持非可微、复杂功能约束（如最小距离 >3m） |
| **灵活性** | 手工奖励工程成本高 | LLM 自动生成奖励程序，支持任意自然语言任务描述 |
| **可靠性** | LLM 直接生成场景易出错 | 以扩散模型为基础，仅通过 RL 微调，保留多样性与真实性 |
| **扩展性** | 难以应对多任务组合 | 两阶段设计允许统一基础 + 多样任务定制 |

---

## 2. **核心实验方法和设置**

### 📚 数据集
- **3D-FRONT**：包含 6,813 个专业设计的室内场景，作为主训练与测试数据集。
- **3D-FUTURE**：提供 CAD 模型用于渲染最终网格。
- 实验聚焦于 **bedroom 场景子集**。

---

### ⚙️ 实验设置
- **Base Model**：采用连续域版本的 **MiDiffusion** 模型，在 3D-FRONT 上预训练。
- **Fine-tuning 方法**：
  - 使用 **LoRA**（rank=16, α=16）进行参数高效微调。
  - 优化算法：**DDPO**（Denoising Diffusion Policy Optimization）。
  - 采样器：20 步 DDIM。
  - 学习率：1e-5，Adam 优化器。
- **LLM Agent**：使用 **Gemini** 模型生成奖励函数，每 10 个 RL epoch 进行一次 reward reflection。

---

### 📊 评估指标
分为三大类：

#### （1）分布质量（Distribution Quality）
- **FID**（Fréchet Inception Distance）：衡量生成场景与真实数据分布的距离（越低越好）。
- **SCA**（Scene Classification Accuracy）：分类准确率，反映多样性覆盖能力。

#### （2）物理合理性（Physical Plausibility）
- `Colobj ↓`：发生碰撞的对象比例。
- `Colscene ↓`：至少有一个碰撞的场景比例。
- `Rout ↓`：超出房间边界的物体比例。

#### （3）功能性效用（Functional Utility）
- `Rreach (%) ↑`：从起始点可达的对象占比。
- `Rwalkable ↑`：最大连通可行走区域面积 / 总可走区域面积。

> 注：所有指标均在 1,080 个生成样本上计算，部分任务还过滤出满足约束的真实子集（记作 3D-FRONT\*）用于对比。

---

### 🔁 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **ATISS** | Autoregressive Transformer | 序列化对象放置，依赖数据分布 |
| **MiDiffusion** | Diffusion-based | 当前 SOTA 的混合离散-连续扩散模型 |
| **MiDiffusion + iARCS aug.** | 数据增强版 | 在原始数据 + iARCS 合成数据上重新训练 |

---

## 3. **主要实验结果和性能指标**

### 📈 关键性能数据（来自 Table 1）

| Method | Colobj ↓ | Colscene ↓ | Rout ↓ | Rreach ↑ | Rwalkable ↑ | FID | SCA |
|--------|----------|------------|--------|---------|--------------|-----|-----|
| **ATISS** | 54.12% | 85.96% | 7.32% | 83.57% | 0.780 | 1.39 | 67.23% |
| **MiDiffusion** | 52.67% | 81.67% | 5.89% | 85.7% | 0.806 | 1.34 | 65.99% |
| **iARCS (Ours)** | **40.45%** | **64.63%** | **3.04%** | **87.82%** | **0.861** | 1.60 | **68.51%** |

✅ 结论：iARCS 显著提升了物理合理性和功能性指标，尽管 FID 略高（因原始数据含噪声），但 SCA 更优，说明**多样性更好**。

---

### 🔍 数据增强效果（Table 2）

| Method | Colobj ↓ | Colscene ↓ | Rout ↓ | Rreach ↑ | Rwalkable ↑ | FID | SCA |
|--------|----------|------------|--------|---------|--------------|-----|-----|
| **MiDiffusion (3D-FRONT)** | 52.67% | 81.67% | 5.89% | 85.7% | 0.806 | 1.34 | 66.00% |
| **+ iARCS aug.** | **41.49%** | **63.61%** | **3.12%** | **92.52%** | **0.8272** | 1.34 | 66.10% |

✅ 结论：仅用 iARCS 生成的 4,000 个合成数据进行再训练，即可显著提升原模型的**物理与功能性能**，同时保持 FID 和 SCA 不变 → 表明 iARCS 是一个有效的 **synthetic data engine**。

---

### 🎯 任务特定约束优化（Table 3）
针对两个典型任务测试：

#### 任务 1：“Robot grasping scene with all supports ≤1.0m vertical reach”
- iARCS 达到更低的 `FID=3.144` vs. `4.36`（3D-FRONT\*），更高 `Rreach=86.38%`。
- 表明：在满足严格约束下仍能生成更丰富多样的合法场景。

#### 任务 2：“TV visible from bed for farsighted person (>3m)”
- iARCS 实现 `FID=2.905` vs. `14.49`（3D-FRONT\*），`Rwalkable=0.790`。
- 极大优于仅靠数据筛选的方法，证明其**强约束下的生成能力优势**。

> 💡 关键发现：**iARCS 生成的场景不仅满足约束，而且比真实数据中符合条件的子集更具多样性（更低 FID）！**

---

### 🔬 消融实验（Ablation Study, Table 4）
比较两种训练方式（相同 reward budget）：

| 方法 | Colobj ↓ | Colscene ↓ | Rout ↓ | Rreach ↑ | Rwalkable ↑ | FID | SCA |
|------|----------|------------|--------|---------|-------------|-----|-----|
| **Single-stage** | 62.17% | 95.83% | 16.07% | 64.66% | 0.679 | 1.96 | 78.69% |
| **Two-stage (Ours)** | **38.79%** | **70.04%** | **5.84%** | **80.37%** | **0.744** | **1.75** | **72.38%** |

✅ 结论：**两阶段训练显著优于单阶段联合优化**，验证了先打好物理基础再引入任务约束的重要性。

---

## 4. **关键结论和发现**

### ✅ 主要发现
1. **iARCS 成功实现了高保真、高多样性的可控 3D 场景生成**：
   - 在多个任务上超越了纯数据驱动模型和手工规则系统。
   - 生成的数据可用于反向增强原始模型，形成良性循环。

2. **LLM + RL 的 agentic loop 是解决复杂约束建模的有效路径**：
   - LLM 能理解语义意图并转化为可执行 reward code。
   - DDPO 支持对非可微目标进行端到端优化。
   - Reward reflection 机制有效缓解 reward hacking 和训练不稳定。

3. **两阶段训练至关重要**：
   - 先优化 universal rewards 可避免模型在 fine-tuning 中退化。
   - 为后续任务特定优化提供了稳定、高质量的基础策略。

4. **iARCS 是一个强大的 synthetic data engine**：
   - 不仅可用于编辑，更能生成高质量合成数据用于训练更强的生成器。

---

### ⚠️ 局限性
1. **LLM 生成奖励的质量依赖 prompt 清晰度**：
   - 模糊或歧义的指令可能导致错误或不完整的约束分解。
2. **计算开销较大**：
   - 多轮 RL 训练 + LLM 调用 + reward evaluation 导致整体耗时较长。
3. **当前局限于 3D-FRONT 分布内场景**：
   - 泛化到其他建筑风格或室外场景尚未验证。

---

### 🔮 未来工作方向
1. **扩展至更广的场景分布**（如办公室、厨房、户外）。
2. **引入 4D 动态生成**（时间维度，模拟物体移动与交互）。
3. **减少对 LLM 的依赖**：探索 reward 函数的自动修复与迁移机制。
4. **应用于具身 AI 的闭环训练系统**：将生成场景直接用于机器人策略训练与反馈迭代。

---

> 📌 **一句话总结**：  
> **iARCS 开创性地将 LLM 的语义理解能力与 DDPO 的非可微优化能力相结合，通过两阶段 agentic RL 框架，实现了既符合功能约束又保持高质量分布的可控 3D 场景生成，并可作为强大合成数据引擎进一步提升基础模型性能。**

</details>

---

### 11. [CircuitSteer: Geometrically Aligned Multi-Layer Steering via Sparse Autoencoder Circuits](https://arxiv.org/abs/2608.05732)

**Authors**: Mehrshad Saadatinia, Parsa Razmara, Ardalan Aryashad, Ali Abbasi, Seyedarmin Azizi  
**Category**: cs.LG  
**Published**: 2026-08-07  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.05732v1  

#### Abstract
Controlling the behavior of large language models (LLMs) remains a critical challenge for AI alignment. Existing steering methods, such as Contrastive Activation Addition (CAA), typically rely on fixed single-layer interventions derived from aggregate activation differences. These methods impose a s...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：CircuitSteer: Geometrically Aligned Multi-Layer Steering via Sparse Autoencoder Circuits**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
大型语言模型（LLMs）的行为控制是AI对齐（AI alignment）中的核心挑战。现有的推理时干预方法（如 Contrastive Activation Addition, CAA）通常依赖于**单层**、**静态**的激活差异向量进行干预，存在以下问题：
- **语义混杂**：在残差流（residual stream）中，多个概念被叠加编码（superposition），单层干预会同时影响多个无关特征。
- **行为不一致**：高层语义概念是跨层分布并逐步演化的，单点干预无法持续控制多层语义轨迹。
- **流畅性破坏**：多层干预若未对齐，会导致**破坏性干扰**（destructive interference），引发文本退化或“过干预”（over-steering）。

### **提出的新方法与创新**
本文提出了 **CircuitSteer**，一种基于 **Sparse Autoencoders (SAEs)** 的**多层几何对齐干预框架**，其核心创新如下：

#### **1. 构建跨层特征流电路（Feature Flow Circuit）**
- 利用 SAE 将残差流分解为稀疏、单义（monosemantic）的特征。
- 通过两个联合标准识别跨层语义子电路：
  - **共激活（Co-activation）**：同一输入下相邻层特征均被激活。
  - **几何对齐（Geometric Alignment）**：要求两层特征的解码器方向（decoder directions）具有高余弦相似度（cosine similarity），避免方向冲突。

#### **2. 多点协同干预（Multi-Point Intervention）**
- 从选中的子电路中合成**稠密干预向量**（dense steering vectors），并在每个相关层施加干预。
- 干预向量由参与电路的所有特征的解码器方向平均得到，确保干预强度稳定。

#### **3. 几何对齐作为必要条件**
- 强调**几何对齐**是实现多层稳定干预的关键，而非可选项。消融实验证明，移除该条件将导致流畅性崩溃。

### **相比现有方法的优势**
| 维度 | CircuitSteer | 传统方法（如 CAA, RepE） |
|------|-------------|------------------------|
| **干预粒度** | 特征级（feature-level） | 残差流级（residual-level） |
| **干预范围** | 跨层、多点 | 单层或非对齐多层 |
| **语义精确性** | 高（针对特定行为电路） | 低（混合多个概念） |
| **流畅性保持** | ✅ 严格保持（PPL ≈ 1.0） | ❌ 易崩溃（PPL >> 1.5） |
| **覆盖能力** | ✅ 对复杂行为（如 sycophancy, refusal）有效 | ❌ 在复杂行为上失效 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
在四个任务上评估，涵盖不同行为类型：
| 数据集 | 任务类型 | 描述 |
|-------|--------|------|
| **RTP (RealToxicityPrompts)** | 毒性控制 | 抑制有毒续写 |
| **Jigsaw/Civil Comments** | 毒性控制 | 抑制仇恨言论 |
| **Emotion** | 情感控制 | 抑制愤怒，保留喜悦 |
| **Sycophancy** | 社交奉承 | 抑制迎合用户偏见的回答 |

此外，在 **AdvBench** 上测试**拒绝行为**（refusal）的干预。

### **模型**
- **Gemma-2-2B**（26层）
- **Llama-3.1-8B-Instruct**（32层）
- 使用公开的 SAE 套件：**Gemma-Scope** 和 **Llama-Scope**（via SAELens）

### **评估指标**
| 指标 | 定义 |
|------|------|
| **Behavioral Reduction (△)** | 行为得分下降量（如毒性概率、奉承倾向等） |
| **Normalized Perplexity (PPL↓)** | 干预后PPL / 原始PPL，1.0表示无退化 |
| **有效性窗口** | 仅考虑 `0.01 ≤ PPL ≤ 1.5` 且 `△ > 0.02` 的配置 |

### **基线方法对比**
共比较 **8种基线**：
- **Prompt**：系统提示词引导
- **CAA (Contrastive Activation Addition)**：单层与多层版本
- **RepE (Representation Engineering)**
- **ITI (Inference-Time Intervention)**
- **LoReFT**：监督微调基线
- **SpARE / SAE-SSV**：基于SAE的单层干预

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table 1）**
在所有 **8个模型-数据集组合** 中，**CircuitSteer 是唯一一个在所有配置下都存在有效操作点的方法**。

| 方法 | 成功配置数 / 8 | 最高△（部分） |
|------|---------------|-------------|
| **CircuitSteer** | ✅ **8/8** | RTP: 0.128 (Llama), Sycophancy: 0.156 (Llama) |
| ITI | 5/8 | Sycophancy: 0.356 (Llama) |
| CAA (multi-layer) | 4/8 | —— |
| RepE | 4/8 | —— |
| 其他 | ≤3/8 | —— |

> 注：“——”表示在有效PPL窗口内无显著行为减少。

### **与基线方法的对比结果**
- **在 Sycophancy 任务上**：
  - **Gemma-2-2B**：所有基线方法均失败（△≈0 或负值），而 **CircuitSteer 达到 △=0.057**。
  - **Llama-3.1-8B**：CircuitSteer 是少数能实现有意义减少的方法之一。
- **在 Refusal 任务上（AdvBench）**：
  - **CircuitSteer (λ=-3)**：拒绝率从 **89% → 0%**，PPL=35.7（仍连贯）。
  - **CAA (λ=-3)**：拒绝率 **92%**（反而上升），几乎无效果。

### **消融实验结果（Table 2）**
验证了三个核心组件的必要性（在 Gemma 和 Llama 上）：

| 变体 | 关键改动 | 结果 |
|------|--------|------|
| **NOGEO** | 移除几何对齐 | 流畅性下降，△降低 |
| **NEGATIVEALIGN** | 使用反向对齐特征 | △显著下降，尤其在 Sycophancy |
| **RANDOMSAE** | 使用随机投影替代SAE | 效果崩溃，无有效点 |
| **SINGLELAYER** | 限制为单层电路 | 在 Sycophancy 上表现接近，但失去多层优势 |

> **结论**：几何对齐和多层协调对复杂行为（如 sycophancy）至关重要。

### **其他重要实验发现**
- **阈值鲁棒性**（Table 12）：超参数（Tact, Tsim, Tdiff）在合理范围内变化时，性能稳定。
- **泛化能力**（Table 11）：在更大的 **Qwen3.5-27B** 模型上，无需调参即取得类似效果，证明方法可扩展。
- **基础能力保留**（Table 15）：在强干预（λ=-3）下，MMLU 和 GSM8K 准确率下降仅约 **0.03**，表明核心知识和推理能力未受损。

---

## **4. 关键结论和发现**

### **主要发现**
1. **多层干预必须几何对齐**：  
   未经对齐的多层干预会导致破坏性干扰，而**几何对齐是实现稳定、流畅干预的必要条件**。

2. **行为是跨层电路实现的**：  
   高层语义行为（如拒绝、奉承）由分布在多个层的**特征流电路**实现，需协同干预。

3. **CircuitSteer 是唯一全覆盖方法**：  
   在所有测试任务和模型上，它是**唯一能始终在保持流畅性的同时实现行为控制的方法**，尤其在复杂行为（sycophancy, refusal）上远超现有方法。

4. **干预可解释且透明**：  
   所构建的电路可直接可视化和分析，实现了从“黑盒干预”到“白盒电路编辑”的转变。

### **方法的局限性**
- **依赖高质量SAE**：需要预先训练好的、覆盖多层的 SAE，目前并非所有模型都有可用套件。
- **离线计算开销**：电路发现为一次性离线过程，虽不影响推理延迟，但需大量计算资源提取特征。
- **对对比集敏感**：在生成式决策任务（如 sycophancy）中，若对比集格式不匹配，可能导致电路发现失败（见 Table 13）。

### **未来工作方向**
- **自动化电路发现**：探索更高效、自适应的电路构建算法。
- **动态干预调度**：根据输入动态选择干预层数和强度。
- **防御机制设计**：基于 CircuitSteer 的洞察，开发更鲁棒的对齐方法，抵御此类细粒度干预。
- **扩展至多模态模型**：将电路思想应用于 VLMs 或多模态基础模型。

---

> **代码地址**：[https://github.com/mehrshad-sdtn/CircuitSteer](https://github.com/mehrshad-sdtn/CircuitSteer)

</details>

---

### 12. [SkillHEX: Improving Agent Skills via Hypothesis-Driven Autonomous Exploration and Exploitation](https://arxiv.org/abs/2608.05628)

**Authors**: Yuru Feng, Yaoqi Chen, Beidi Zhao, Qianxi Zhang, Xinjiang Wang, Jianan Lu, Zhirui Wang, Shusen Xu, Zengzhong Li, Qi Chen  
**Category**: cs.AI  
**Published**: 2026-08-07  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.05628v1  

#### Abstract
Although agent skills equip LLMs with reusable procedural knowledge, manual maintenance suffers from high costs, unscalability, and misalignment. Real-world deployments thus require autonomous, on-demand skill evolution at test time, constrained by limited interaction budgets and a lack of training ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：SkillHEX: Improving Agent Skills via Hypothesis-Driven Autonomous Exploration and Exploitation**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前 LLM 驱动的智能体（agent）虽然具备执行能力，但其 **agent skills**（可复用的程序化知识模块）通常依赖人工维护，存在以下问题：
- **高成本、不可扩展**：手动构建和调试技能代价高昂。
- **测试时适应性差**：在真实部署中，缺乏训练/验证集，仅能在有限交互预算下进行试错，面临**稀疏奖励（sparse reward）挑战**。
- **诊断模糊导致误判**：一次失败可能由多种潜在原因引起，传统方法容易因早期误诊陷入“**利用陷阱（exploitation trap）**”，即在错误路径上持续迭代而耗尽预算。

### **提出的新方法与创新思路**
作者提出 **SkillHEX**，一个闭环框架，通过**假设驱动的自主探索与利用**来实现技能的自我演化。其核心创新包括：

#### **(1) Hypothesis-Driven Self-Verification（假设驱动的自验证）**
- 将失败归因于具体的、可证伪的**故障假设（falsifiable failure hypotheses）**。
- 将这些假设转化为**可执行的测试用例（executable tests）**，并在缓存的输出上回放，生成密集的诊断证据（dense diagnostic evidence），无需额外环境交互。
- 这些证据作为“语义梯度（semantic gradients）”指导后续修改。

#### **(2) Evidence-Guided Tree Search（证据引导的树搜索）**
- 构建一个**持久化的技能补丁树（persistent skill-patch tree）**，每个节点代表一个技能版本。
- 搜索过程动态平衡**探索（exploration）** 与 **利用（exploitation）**：
  - 利用：优先评估证据支持的分支。
  - 探索：保留低排名但合理的替代路径，避免过早收敛。
- 使用类似 **PUCT** 的选择机制，结合**最大备份值（max-backup）** 和基于反思排序的**策略先验（policy prior）**。

### **相比现有方法的优势**
| 方面 | 传统方法（如 CoEvoSkills, SkillRevise） | SkillHEX |
|------|----------------------------------------|---------|
| **修订机制** | 贪心地原地更新（in-place refinement） | 维护多条并行分支的树结构 |
| **反馈密度** | 依赖稀疏的最终奖励 | 通过自验证生成密集诊断信号 |
| **容错性** | 一旦误诊即陷入局部最优 | 可回溯到其他分支，跳出陷阱 |
| **效率** | 易在无效路径上浪费预算 | 更高效利用有限尝试 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **SkillsBench**（Li et al., 2026a）：包含 **87 个任务**，覆盖 8 个领域：
  - Software Engineering, Cybersecurity, Natural Science
  - Finance & Economics, Office & White Collar
  - Media & Content Production, Industrial & Physical Systems
  - Mathematics & OR
- 每个任务运行在独立的 Docker 容器中，配有确定性验证器（verifier），仅返回二元成功信号（pass/fail）。

### **实验设置**
- **任务定义**：$ T = (I, D, E) $，其中指令 $ I $、公开数据 $ D $、执行环境 $ E $ 已知，但成功标准隐藏。
- **初始技能**：所有自演化方法均从相同的 **LLM-generated skill**（由 Skill Creator 生成）开始。
- **迭代预算**：最多 **5 次执行尝试（five-iteration budget）**。
- **模型后端**：使用两种主流 LLM：
  - **GPT-5.3-Codex**（OpenAI）
  - **Claude Opus 4.7**（Anthropic）

### **评估指标**
- **主指标**：**task-macro pass rate**（任务宏平均通过率）
- 报告 **pass@5**（5 次迭代后的最终通过率）
- 相对提升以 **percentage point gain (pp)** 表示
- 所有实验重复 **3 次取均值 ± 标准差**

### **基线方法对比**
| 类型 | 方法 | 描述 |
|------|------|------|
| **静态基线** | `No Skill` | 不安装任何技能 |
| | `Skill Creator` | 使用原始 LLM 生成的技能（未优化） |
| | `Human` | 使用人类专家编写的技能 |
| **自演化基线** | `CoEvoSkills`（Zhang et al., 2026b） | 协同演化技能与代理验证器 |
| | `SkillRevise`（Liu et al., 2026b） | 基于执行轨迹的顺序修订 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 方法 | GPT-5.3-Codex | Claude Opus 4.7 |
|------|----------------|------------------|
| **SkillHEX (Ours)** | **55.9%** | **57.9%** |
| CoEvoSkills | 46.4% | 49.4% |
| SkillRevise | 44.4% | 44.8% |
| Human | 52.9% | 54.4% |
| Skill Creator | 39.5% | 39.1% |
| No Skill | 33.3% | 34.1% |

> ✅ SkillHEX 在两个 backbone 上均显著优于最强基线（CoEvoSkills），分别提升 **9.5 pp** 和 **8.5 pp**，且**超越人类水平**。

### **与基线方法的对比结果**
- **全面领先**：在所有 8 个领域中，SkillHEX 均优于所有基线。
- **尤其擅长复杂流程任务**：
  - 在 **Mathematics & OR** 领域，相对人类技能提升高达 **33.3–37.5 pp**。
  - 在 **Office & White Collar** 提升 **19.0–21.4 pp**。
- **在知识密集型领域仍有差距**：
  - 如 **Natural Science** 和 **Finance & Economics**，仍略低于人类技能，表明存在**知识获取瓶颈**。

### **消融实验结果（Ablation Studies）**
| 方法 | Pass Rate (GPT-5.3-Codex) | Drop (pp) |
|------|----------------------------|----------|
| **SkillHEX (full)** | **55.9%** | 0.0 |
| w/o self-verifier | 44.8% | ↓11.1 |
| w/o skill patch tree | 49.1% | ↓6.8 |

- **移除 self-verifier** 导致性能大幅下降（↓11.1 pp），说明**密集诊断信号是性能提升的关键**。
- **替换为 in-place refinement** 也导致显著下降（↓6.8 pp），证明**树结构对避免局部最优至关重要**。
- 即使没有树结构，只要有 self-verifier，仍优于 CoEvoSkills，体现其**假设驱动设计的优越性**。

### **Token 成本分析**
| 方法 | 总 Token 数量（GPT-5.3-Codex） |
|------|------------------------------|
| CoEvoSkills | 2874.3K |
| **SkillHEX (full)** | **2356.1K** |
| w/o self-verifier | 1161.4K |
| w/o skill patch tree | 2502.2K |

- SkillHEX 比最强基线 **节省 18.0% 的 token**。
- 使用树搜索反而比 in-place refinement **更高效**，因其能通过历史证据快速淘汰劣质路径。

---

## **4. 关键结论和发现**

### **主要发现**
1. **稀疏奖励下的有效学习需要主动获取密集信号**：
   - 仅靠最终成败无法支撑有效信用分配。
   - SkillHEX 通过 **hypothesis-driven self-verification** 将模糊失败转化为可操作的诊断证据，极大提升了学习效率。

2. **探索与利用的平衡至关重要**：
   - 贪心策略易陷入“利用陷阱”。
   - **persistent patch tree + evidence-guided search** 允许回溯和分支切换，是突破局部最优的关键。

3. **强初始化加速收敛**：
   - 从人类技能出发比从零开始或从 LLM 技能出发效果更好。
   - 人类提供强先验，SkillHEX 将其细化为精确的任务级规范（如 artifact contracts, numerical conventions）。

4. **方法具有通用性和鲁棒性**：
   - 在不同 LLM backbone 上表现一致。
   - 消融实验显示各组件贡献互补且稳健。

### **局限性**
- **知识获取瓶颈**：在高度依赖专业知识的领域（如 Natural Science），性能受限于 LLM 自身的知识边界，难以仅从执行反馈中推断缺失事实。
- **假设质量依赖 LLM 推理能力**：若 LLM 生成错误或无关假设，会影响整个流程。
- **计算开销仍较高**：尽管 token 更省，但整体流程涉及多次 LLM 调用（reflection, verification, search），延迟较高。

### **未来工作方向**
- 结合外部知识检索（Knowledge Retrieval）以缓解知识瓶颈。
- 动态调整 hypothesis generation 策略，提高假设质量。
- 将 SkillHEX 应用于多技能协同演化或多智能体系统。
- 探索更高效的搜索策略（如基于 learned value model 的引导）。

---

> **总结**：SkillHEX 通过 **hypothesis-driven self-verification** 和 **evidence-guided tree search**，在测试时有限预算下实现了 agent skills 的高效自主演化。其实验证明，**主动生成诊断证据** 和 **保留探索路径** 是克服稀疏奖励与模糊归因的核心机制，为构建真正自主进化的智能体提供了重要范式。

</details>

---

### 13. [Integrating Implicit and Explicit Relational Biases through Graph-Based Multiple Instance Learning: A Case Study in Skin Lesion Diagnosis](https://arxiv.org/abs/2608.06037)

**Authors**: Rafa{\l} Buler (Gda\'nsk University of Technology), Jakub Buler (Gda\'nsk University of Technology), Maciej Bobowicz (Medical University of Gda\'nsk), Micha{\l} Grochowski (Gda\'nsk University of Technology)  
**Category**: cs.AI  
**Published**: 2026-08-07  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.06037v1  

#### Abstract
Relational inductive biases are essential for capturing structural dependencies among data. This study investigates a dual-level relational framework for image classification, bridging the gap between implicit representation learning and explicit structural modelling. We begin by establishing a base...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Integrating Implicit and Explicit Relational Biases through Graph-Based Multiple Instance Learning: A Case Study in Skin Lesion Diagnosis

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该研究旨在解决**医学图像分类中如何有效建模局部区域（如图像块 patches）之间的关系**这一关键挑战。传统 CNN 主要依赖卷积操作隐式捕捉空间依赖，而 patch-based 方法通常将 patch 视为独立实例处理，忽略了它们之间的显式结构关系。此外，在标注数据稀缺的医疗场景下，如何结合自监督学习（SSL）与图神经网络（GNN）来提升模型性能仍不明确。

### 🚀 提出的新方法/新思路
本文提出了一种**双层级 relational framework**，融合了：
- **Implicit relational bias**：通过基于 Convolutional Masked AutoEncoder (ConvMAE) 的自监督预训练，让模型在重建 masked patches 的过程中**隐式学习 patch 间的局部上下文关系**。
- **Explicit relational bias**：将图像表示为图结构（graph），利用 Graph Neural Networks（GNN）进行 message passing，**显式建模 patch 之间的拓扑关系**。

具体流程如下：
1. 使用冻结的 ConvMAE 编码器提取 14×14 = 196 个 non-overlapping patch embeddings；
2. 将这些 patch embeddings 构造成图（graph），采用三种构造策略：
   - Random graph（随机连接）
   - Grid graph（基于原始空间位置的邻接连接）
   - k-Nearest Neighbour (kNN) graph（基于特征空间距离）
3. 在图上应用 GNN（如 GCN、GAT）进行多层 message passing；
4. 最后使用 attention-based MIL pooling 聚合节点表示并分类。

> 这种“先隐式学习 + 后显式建模”的两阶段设计是本工作的核心创新。

### 🔍 相比现有方法的优势
| 对比维度 | 优势说明 |
|--------|---------|
| **与标准 CNN 比较** | 引入 patch-level 结构建模能力，突破局部感受野限制，更适合皮肤病变等需整合多区域视觉线索的任务。 |
| **与纯 SSL 或纯 GNN 方法比较** | 不仅利用 SSL 学习高质量 patch 表示，还进一步通过 GNN 显式增强其结构性，实现互补增益。 |
| **与简单 MIL 比较** | 引入图结构打破了 patch 独立假设，能更好地捕获语义相关性与空间布局信息。 |
| **轻量高效 vs ensemble models** | 单一模型即可达到接近甚至超越复杂集成模型（ensemble of 90 models）的性能，具备更高性价比。 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **ISIC-2018 Challenge Task 3**  
  - 类别数：7类  
  - 训练集：10,015 图像  
  - 测试集：1,514 图像  

- **ISIC-2019**  
  - 类别数：8类（含一个“unknown”类别）  
  - 训练集：25,331 图像  
  - 测试集：8,238 图像  
  - “unknown”样本（99例）被排除以避免分布外干扰

> 自监督预训练阶段联合使用两个数据集的所有未标记图像；下游任务中分别独立评估。

### ⚙️ 实验设置
- **输入预处理**：中心裁剪为正方形 → resize 到 `224×224` → 使用 ImageNet 统计值归一化
- **patch 分割**：划分为 `16×16` 像素的小块 → 得到 `14×14=196` 个 patches，每个 embedding 维度为 768
- **SSL 预训练**：使用 ConvMAE，masking ratio 设为 0.5（经线性探针验证最优）
- **下游分类器**：Attention-based Multiple Instance Learning (AMIL)，支持 permutation invariance
- **GNN 架构变体**：测试了 GCN、GAT 和 Graph Transformer
- **图构建策略对比**：
  - Random graph (`r=1~4`)
  - Grid graph（4- 或 8-邻域连接）
  - kNN graph（k=1~4）

### 📊 评估指标
- **Balanced Accuracy (BAcc)**：为主要评价指标，用于缓解类别不平衡影响
- **五折交叉验证（5-fold CV）**：在 ISIC-2018 上执行，报告均值 ± 标准差
- 所有超参数统一调优协议（learning rate, optimizer, dropout, GNN depth 等）

### 🆚 基线方法
| 基线模型 | 描述 |
|-------|------|
| EfficientNet-B3 | 全监督端到端训练的 CNN 基线 |
| Base AMIL | 使用 ConvMAE 提取 patch embeddings + attention pooling，无图结构 |
| Ensemble Models | 多模型集成方法（来自文献对比）|

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（Balanced Accuracy）

#### ✅ ISIC-2018 Test Set 结果
| Model | Relational Bias | BAcc [%] |
|------|------------------|----------|
| EfficientNet-B3 | Pixel-level (P) | 76.17 ± 0.89 |
| Base AMIL | P + Implicit (I) | 77.12 ± 1.49 |
| Graph AMIL (Random, r=2) | P+I+ERandom | 78.24 ± 0.86 |
| **Graph AMIL (Grid + GAT)** | **P+I+EGrid** | **79.27 ± 1.38** ✅ |
| Ensemble (90 models) | – | 86.20 |

> **提升显著**：从 baseline 的 76.17% 提升至 **79.27%**，相对提高约 4%，且优于部分小规模集成模型（如 25-model ensemble）。

#### ✅ ISIC-2019 Test Set 结果
| Model | Relational Bias | BAcc [%] |
|------|------------------|----------|
| Base AMIL | P + I | 59.84 ± 1.50 |
| **Graph AMIL (kNN + GAT, k=4)** | **P+I+EKNN** | **60.67 ± 0.68** ✅ |
| Ensemble-Top-2 (Leaderboard) | – | 60.70 |
| Ensemble-Top-1 (Best) | – | 63.60 |

> 所提单模型表现已**逼近顶级集成方法**，远超非集成前五方案（最高仅 56.90%）。

### 🔬 消融实验结果
| 设置 | BAcc [%] (ISIC-2018) | 分析 |
|-----|--------------------|------|
| Base AMIL | 77.12 | 仅有隐式关系建模 |
| + Random Graph (r=2) | 78.24 | 引入任意连接即有增益 |
| + Grid Graph (GAT) | 79.27 | **空间结构最有效** |
| + Frozen GNN Weights | 34.28 ❌ | 性能大幅下降，表明**learned message passing 至关重要**，而非单纯增加容量 |

> ➤ 显式关系建模的有效性依赖于可训练的 GNN 参数，固定权重导致灾难性退化。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **隐式 + 显式 relational bias 可协同增效**  
   自监督学习（ConvMAE）提供了富含局部上下文的 patch 表示（implicit bias），而 GNN 引入的空间或语义图结构（explicit bias）进一步提升了全局理解能力。

2. **图结构的设计至关重要**  
   - **Grid graph > kNN > Random graph**：原始空间邻接关系比特征相似性更能反映皮肤病变的结构规律。
   - 表明**空间先验知识对皮肤病诊断具有重要意义**。

3. **GNN 的可学习性是关键**  
   冻结 GNN 层导致性能骤降（从 79.27% → 34.28%），证明 message passing 必须经过训练才能有效传递有意义的信息。

4. **高性能无需大规模集成**  
   单一模型即可达到接近 ensemble 方法的表现，尤其在 ISIC-2019 上已进入 leaderboard 前两名水平，展现出极强竞争力。

### ⚠️ 方法的局限性
- **计算开销增加**：引入 GNN 增加了推理延迟，可能不利于实时部署。
- **图构建依赖手工设计**：目前 grid/kNN/random 是人为设定，缺乏自动学习最优拓扑的能力。
- **泛化性待进一步验证**：虽声称可用于其他医学领域，但当前仅在皮肤镜图像上验证。

### 🔮 未来工作方向
1. 探索 **dynamic graph construction**，例如通过可学习的 attention 机制自动建立 patch 间连接。
2. 将框架扩展至 **3D 医学影像**（如 MRI、CT）中的病灶分析。
3. 结合 **multi-modal data**（临床文本 + 图像）构建更全面的诊断系统。
4. 开发 **lightweight GNN variants** 以适应边缘设备部署需求。

---

## ✅ 总结
该论文系统地探索了 **implicit 与 explicit relational bias 的融合路径**，提出了一个基于 **Graph-based MIL 的双阶段架构**，在皮肤病变诊断任务上取得了优异成果。其实验严谨、消融充分，揭示了：
> **“好的表示 + 显式的结构建模” > 单独任一方**

不仅推动了医学图像分析的发展，也为构建更具解释性和结构感知能力的深度学习模型提供了重要范式参考。

</details>

---

### 14. [DBLAST: Dependent Block Drafting for Stochastic Speculative Decoding](https://arxiv.org/abs/2608.05448)

**Authors**: Amirmohammad Karimi, Chao Gao, Negar Hassanpour  
**Category**: cs.CL  
**Published**: 2026-08-07  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.05448v1  

#### Abstract
Speculative decoding accelerates large language models' inference by using a lightweight drafter to propose multiple future tokens and a target model to verify them. While recent block and diffusion-style drafters can predict several positions in a single pass, their training and sampling procedures...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：DBLAST: Dependent Block Drafting for Stochastic Speculative Decoding**

---

## **1. 主要贡献和创新点**

### **解决的问题**
当前的 **speculative decoding** 方法（如 DFlash、Medusa）在非贪婪（non-greedy）解码场景下存在性能瓶颈。这些方法通常假设 draft block 中各 token 位置是**条件独立**的（independent block sampling），即每个位置独立预测。然而，在高熵（high-entropy）目标分布中（如创意写作、对话等开放任务），多个合理续写路径并存，这种独立性假设会导致生成的 draft block 缺乏内部一致性，从而降低**验证接受长度**（accepted length），削弱加速效果。

### **提出的新方法：DBLAST**
论文提出了 **DBLAST**（Dependent Block LATent Sampler with Training on Accepted Length），其核心创新包括两点：

#### **(1) 依赖性块采样（Dependent Block Sampling）**
- 引入一个**低秩潜在混合模型**（low-rank latent mixture）通过一个**分类潜变量** $ z \in \{1,\dots,K\} $ 来建模 block 内部 token 之间的依赖关系。
- 在每个潜类别 $ z $ 下，并行预测整个 block 的 token，不同类别代表不同的连贯续写模式。
- 通过边际化 $ z $ 得到联合分布，使 block 内部 token 形成**语义一致的路径**，而非孤立预测。

#### **(2) 面向接受长度的训练目标（Loss on Accepted Length, AL）**
- 不再使用传统的 **Negative Log-Likelihood (NLL)**，而是设计了一个基于**期望接受长度**（expected accepted length）的代理损失函数。
- 该损失结合了：
  - **重要性比**（importance ratio）反映 proposal 与 target 分布匹配度；
  - **前缀接受信号**（prefix acceptance）模拟 speculative verification 的顺序行为；
  - **阈值截断机制**（threshold truncation）稳定训练初期梯度。

> ✅ **优势总结**：
> - 保持了 block speculative decoding 的**单次并行推理效率**；
> - 显著提升在**高熵解码场景下的接受长度**；
> - 损失函数更贴近 speculative decoding 的实际验证过程。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **训练数据**：来自 `Tulu3 SFT mixture` 的提示词，由目标模型生成响应用于训练 drafter。
- **评估基准**：
  - **GSM8K**：数学推理
  - **MT-Bench**：多轮对话质量
  - **HumanEval**：代码生成
  - **Creative Writing**：使用 EQ-Bench Creative Writing Benchmark v3 测试多样性任务

### **实验设置**
- **目标模型**：
  - Qwen3-4B 和 Qwen3-8B
- **drafter 架构基础**：基于 DFlash 的 block diffusion drafter
- **训练配置**：
  - 温度（temperature）：0.7 / 1.0 / 1.5
  - top-p：0.8 / 0.95
  - top-k：20
  - 块长度（block length）：15 tokens
  - 训练周期：1 epoch
- **推理设置**：
  - 最大生成 256 tokens
  - 使用 greedy branch decoding：先采样类别 $ z $，然后在其分支上贪心解码
  - 类别温度 $ Z_T $ 在校准集上调优后固定

### **评估指标**
- **主要指标**：**macro-average accepted length**（平均接受长度）
  - 衡量 speculative decoding 的有效加速能力
- **辅助分析**：按“目标块早期确定性”（early determinism）分组统计接受长度

### **基线方法对比**
| 方法 | 是否依赖 | 训练目标 |
|------|--------|----------|
| **DFlash** | 否（K=1） | NLL |
| **DFlash + AL** | 否（K=1） | AL loss |
| **DFlash + DS** | 是（K=4） | NLL |
| **DBLAST** | 是（K=4） | AL loss |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table 2）**

| 模型 | 设置 | DFlash | DBLAST | 提升幅度 |
|------|------|--------|--------|---------|
| Qwen3-4B | T=0.7, p=0.8 (高确定性) | 4.55 | 4.81 | +5.8% |
| Qwen3-4B | T=1.5, p=0.95 (低确定性) | 4.01 | 4.38 | +9.3% |
| Qwen3-8B | T=0.7, p=0.8 (高确定性) | 4.55 | 4.81 | +5.6% |
| Qwen3-8B | T=1.5, p=0.95 (低确定性) | 3.70 | 4.15 | **+12.1%** |

> 🔺 **最高提升达 12.1%**，且**增益随目标分布熵增加而增大**。

### **与基线对比结果**
- **DBLAST > 所有变体**：在所有任务、模型和解码设置下均取得最佳接受长度。
- **消融实验证明两个组件互补**：
  - **仅用 AL 损失（K=1）**：平均提升约 5%
  - **仅用依赖建模（K=4 + NLL）**：平均提升约 6–7%
  - **两者结合（DBLAST）**：提升达 9–12%，说明二者协同增强。

### **消融实验结果**

#### **(1) 潜类别数量 $ K $ 的影响（Table 3）**
| K | NLL | AL |
|----|-----|-----|
| 1 | 4.00 | 4.17 |
| 2 | 4.10 | 4.31 |
| 4 | 4.10 | **4.39** |
| 8 | 4.13 | 4.41 |

- AL 更能利用额外分支容量，$ K=4 $ 即可获得大部分收益。

#### **(2) 推理时类别温度 $ Z_T $ 的影响（Figure 2）**
- AL 训练的 drafter 在所有 $ Z_T $ 下表现优于 NLL。
- 最佳 $ Z_T $ 随目标熵上升而提高 → 支持“draft stochasticity 应匹配 target stochasticity”的观点。
- 过高的 $ Z_T $ 反而下降 → **需校准而非任意增加噪声**。

#### **(3) AL 损失构造消融（Table 4）**
| 构造方式 | K=1 | K=4 |
|--------|-----|-----|
| 整个 block 优化 | 4.73 | 4.98 |
| 仅一个前缀 | 4.87 | 5.18 |
| **所有保留前缀求和（T=0.1）** | **4.99** | **5.33** |

- “sum all retained prefixes” 效果最好，提供更密集监督信号。

#### **(4) 早期确定性分组分析（Figure 1）**
- 当目标块早期确定性越低（即越不确定），独立采样的接受长度急剧下降。
- DBLAST 在最不确定区域（bin 1）相对提升高达 **54.2%**，验证其在多样化场景中的优越性。

---

## **4. 关键结论和发现**

### **主要发现**
1. **独立块采样在非贪婪解码中存在根本性缺陷**：
   - 忽略 block 内部依赖导致 proposal 虽然 marginally 准确，但路径不连贯。
2. **DBLAST 成功缓解该问题**：
   - 通过 latent mixture 建模 coherent 续写路径；
   - AL 损失直接优化 speculative verification 的核心指标——接受长度。
3. **两大组件相辅相成**：
   - 依赖建模提供结构表达力；
   - AL 损失引导训练朝实际加速目标优化。
4. **尤其适用于高熵任务**：
   - 如 creative writing、dialogue 等需要多样性的场景，DBLAST 提升最为显著。

### **方法的局限性**
1. **训练与推理 proposal 不一致**：
   - 训练使用 soft latent mixture（可微）；
   - 推理使用 hard greedy branch（高效），二者之间缺乏理论衔接。
2. **AL 损失非无偏估计**：
   - 并非期望接受长度的无偏估计或下界，仅为启发式代理。
3. **训练数据固定为中等熵**：
   - 未探索更高随机性的训练目标块是否进一步提升依赖建模效果。

### **未来工作方向**
- 设计更紧密对齐 speculative acceptance 的**形式化可证明目标函数**。
- 研究训练与推理 proposal 之间的**一致性理论**。
- 探索在训练阶段引入更多**高熵目标响应**以更好激发依赖建模潜力。
- 将 DBLAST 思路扩展至其他 multi-token prediction 架构（如 Medusa、EAGLE）。

---

> ✅ **总结一句话**：  
> **DBLAST 通过引入 latent mixture 建模 block 内依赖，并采用面向接受长度的训练目标，在保持高效并行 drafting 的同时，显著提升了非贪婪 speculative decoding 的接受长度，尤其在高熵场景下优势明显。**

</details>

---

### 15. [Dynamic Graph Prompting via Topology-Routed Mixed-Curvature Experts](https://arxiv.org/abs/2608.06031)

**Authors**: Quanxin Wang, Xuanting Xie, Bingheng Li, Xingtong Yu, Shuo Wang, Ruiyi Fang, Zhao Kang  
**Category**: cs.LG  
**Published**: 2026-08-07  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.06031v1  

#### Abstract
Dynamic graph prompting freezes a pre-trained temporal backbone and adapts it to label-scarce downstream tasks using lightweight prompts. However, existing methods operate within a single, fixed embedding space. In this work, we reveal that temporal shifts in local clustering and degree heterogeneit...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Dynamic Graph Prompting via Topology-Routed Mixed-Curvature Experts**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前的 **dynamic graph prompting** 方法虽然通过冻结预训练的时序图模型并仅微调轻量级 prompts 来适应标签稀缺的下游任务，但它们普遍假设所有节点-时间实例共享一个固定的嵌入空间（通常是 Euclidean 空间）。这种“固定几何”假设在动态图中存在严重缺陷：

- 动态图的局部拓扑结构随时间演化（如聚类系数下降、度异质性上升），导致最优表示几何（representation geometry）也随之变化。
- 这种现象被作者称为 **geometry under-adaptation** —— 即模型无法跟踪由拓扑漂移引起的曲率变化。

### 🚀 提出的新方法：**CURvPROMPT**
为解决上述问题，作者提出 **CURvPROMPT**，一种基于拓扑路由的混合曲率提示框架，其核心思想是：

- 引入一组 **curvature-diverse Riemannian experts**（双曲、欧氏、球面等不同曲率的专家）构成专家库。
- 设计 **topology-aware gate**，根据每个节点-时间实例的多尺度局部拓扑特征，动态地将其实例路由到最合适的少数几个专家（sparse subset）。
- 每个专家配备可学习的 **expert-specific prompt**，实现对特定几何空间的个性化引导。
- 采用 **soft-to-hard routing** 策略：
  - 预训练阶段使用 **soft Top-K routing** 学习连续的拓扑-几何映射；
  - 下游微调阶段切换为 **hard Top-K routing with uniform weights**，提升参数效率和稳定性。

### 🔍 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **几何适应性** | 支持动态选择最优曲率空间，而非固定单一几何（如 Euclidean-only） |
| **参数效率** | 冻结主干网络与专家，仅微调 prompts 和任务头，适合 few-shot 场景 |
| **表达能力** | 混合曲率 + 拓扑感知路由 → 构建个性化 mixed-curvature 表示 |
| **稳定性** | 软-硬路由转换避免下游稀疏标签下的过拟合 |

---

## 2. 核心实验方法和设置

### 📚 数据集
在四个公开的连续时间动态图数据集上进行评估：

| 数据集 | 类型 | 节点数 | 边数 | 时间跨度 | 任务 |
|-------|------|--------|--------|----------|------|
| **Wikipedia** | 用户-编辑行为 | 9,227 | 157,474 | 30天 | NC, LP |
| **Reddit** | 用户-帖子互动 | 11,000 | 672,447 | 30天 | NC, LP |
| **MOOC** | 学生-课程点击 | 7,144 | 411,749 | 30天 | NC, LP |
| **Genre** | 用户-音乐流派偏好 | 1,505 | 17.8M | 1,500天 | NC, LP |

> 所有数据集均来自 TGB（Temporal Graph Benchmark）或原始文献。

### ⚙️ 实验设置
- **预训练**：使用前 80% 时间段的数据，目标为 temporal link prediction（BCE loss）。
- **下游任务**：
  - **few-shot setting**：从剩余 20% 中采样 30 个事件作为训练集（约 0.01% 数据）。
  - 分为 transductive 和 inductive 两种设定。
- **评估指标**：AUC-ROC (%)，报告 100 次独立任务平均值 ± 标准差（5 个随机种子）。

### 🆚 基线方法对比
共四类 baseline：

1. **传统 DGNNs**  
   `GCN-ROLAND`, `GAT-ROLAND`, `TGAT`, `TGN`, `TREND`, `GraphMixer`

2. **动态图预训练方法**  
   `DDGCL`, `CPDG`

3. **静态图提示学习**（迁移到动态图）  
   `GraphPrompt`, `ProG`

4. **动态图提示方法**（主要对比对象）  
   `TIGPrompt`, `DyGPrompt`（state-of-the-art）

> CURvPROMPT 基于 TGN 主干构建，并与其他 prompt 方法保持公平比较。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（AUC-ROC %）
见下表（摘自原文 Table 1）：

| 方法 | Wiki NC | Reddit NC | MOOC NC | Genre NC | LP-Trans | LP-Indu |
|------|---------|-----------|---------|----------|----------|---------|
| **TGN-DyGPrompt** | 74.47 | 74.00 | 69.06 | 51.97 | 94.33 / 96.82 / 70.17 / 87.02 | 92.22 / 95.69 / 69.77 / 87.63 |
| **CURvPROMPT** | **76.17** | **75.60** | **79.15** | 50.27 | **95.70** / **97.30** / **84.05** / **94.40** | **93.56** / **96.33** / **83.80** / **93.93** |

#### ✅ 性能亮点：
- 在 **8 个 link prediction 设置中全部领先**，最大增益达 **+13.9%**（MOOC transductive vs DyGPrompt）。
- 在 **node classification** 上取得 2/4 最优（Reddit, MOOC），其余接近最优。
- 尤其在 **inductive LP** 上表现稳健，显著优于非 prompt 方法（如在 Genre 上接近随机猜测 vs CURvPROMPT 维持高 AUC）。

---

### 🔍 消融实验结果（Ablation Study）

| 变体 | Reddit LP-Trans ↓ | MOOC LP-Trans ↓ | 说明 |
|------|------------------|------------------|------|
| **Full Model** | 97.30 | 84.05 | 完整模型 |
| w/o Topology | 97.22 | 78.11 | 移除拓扑编码影响较小（Reddit），但在 MOOC 明显下降 |
| w/o Expert Prompt | 95.49 | 77.17 | 专家提示重要，尤其对 LP |
| w/o GE (正则项) | 95.71 | 76.63 | 几何正则化有助于稳定路由 |
| w/o Balance | 93.42 | 72.74 | 负载均衡损失防止专家坍缩 |
| **w/o Expert** | 87.29 | 56.32 | **性能崩溃！** 证明混合曲率专家是核心 |

> ➤ “w/o Expert” 表现接近随机，说明 **multi-curvature expert mixture 是成功的关键**。

---

### 🧪 几何归因分析（Geometry Attribution）

进一步验证是否真的是“混合曲率”而非“MoE 容量”带来收益：

| 配置 | Reddit LP-Trans | MOOC LP-Trans |
|------|------------------|----------------|
| **Mixed Curvature** | 97.30 | 84.05 |
| Single Curvature（统一非欧） | 95.64 | 75.68 |
| Euclidean MoE（全欧氏） | 94.56 | 68.92 |

> ➤ 结果表明：
> - 即使使用非欧几何，若不允许多曲率专业化，仍会损失性能；
> - **混合曲率设计本身带来了额外增益**，不是单纯 MoE 结构的效果。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **拓扑演化引发几何漂移**：  
   动态图中的局部聚类和度方差变化会导致边的 Ollivier-Ricci 曲率分布发生系统性偏移（向负曲率迁移），表明最优表示几何应随之演化。

2. **geometry under-adaptation 是真实存在的瓶颈**：  
   固定几何的 prompting 方法难以捕捉这种动态需求，限制了 few-shot 泛化能力。

3. **CURvPROMPT 成功实现了 geometry-adaptive prompting**：  
   - 拓扑感知门控能够根据局部结构动态分配专家；
   - 实验显示专家使用随时间演变（如 Reddit 上球面专家主导趋势）；
   - 混合曲率表示显著提升了 link prediction 性能。

4. **参数高效且稳定**：  
   soft-to-hard routing 策略使得在极低标签场景下也能稳定复用预训练知识。

---

### ⚠️ 局限性
1. **专家数量与曲率初始化依赖先验设计**：  
   当前未自动学习最优专家配置，需手动设定曲率池（positive/near-zero/negative）。
   
2. **计算开销略高于单空间方法**：  
   尽管只微调 prompts，但维护多个 Riemannian 专家仍有一定内存负担。

3. **在拓扑稳定的图中增益有限**：  
   如 Genre 上 NC 性能平庸，可能因其拓扑较稳定，几何自适应空间小。

---

### 🔮 未来工作方向
1. **自动化专家结构搜索**：联合优化专家数量、曲率值与路由机制。
2. **引入可微分曲率学习**：让每个专家的 curvature 成为可训练变量。
3. **扩展至异构图与文本增强图**：结合 textual prompts 实现 multimodal dynamic prompting。
4. **理论分析拓扑-几何映射关系**：建立更坚实的几何迁移理论基础。

---

## ✅ 总结一句话
> **CURvPROMPT 首次将 topology-aware routing 与 mixed-curvature Riemannian experts 相结合，在动态图提示学习中实现了几何自适应，显著提升了 few-shot link prediction 与 node classification 的性能，验证了“representation geometry 应随拓扑演化”的核心理念。**

</details>

---

### 16. [LLM Inference Under Bursty Workload Distribution: Modifying the WAIT Algorithm](https://arxiv.org/abs/2608.06135)

**Authors**: Anjali Gangadhar Katageria, Shobha Rani, Raghu Nandan Sengupta  
**Category**: cs.LG  
**Published**: 2026-08-07  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.06135v1  

#### Abstract
Large Language Models (LLMs) such as ChatGPT and Claude are widely used for information retrieval and problem-solving. Recent work has focused on improving scheduling algorithms to boost throughput while maintaining low latency. However, these approaches often assume Poisson request arrivals with co...

---

### 17. [Threshold-Based Early Stopping of Accumulations in Neural Networks with Binary Activation](https://arxiv.org/abs/2608.06177)

**Authors**: Quentin Luquet de Saint-Germain, Massil Ait Abdeslam, Jean Pierre David  
**Category**: cs.LG  
**Published**: 2026-08-07  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.06177v1  

#### Abstract
Binary neural networks are very attractive for constrained deployment, enabling small footprint and low-power inference. For binary activations, the dot products become sign-controlled additions or subtractions, but the number of operations is unchanged. Indeed, every neuron or output channel still ...

---

### 18. [Adaptive Arena-based Contestable Argumentative Network-of-Experts for Open-Ended Care Plan Coordination](https://arxiv.org/abs/2608.05391)

**Authors**: Truong Thanh Hung Nguyen, Hoang-Loc Cao, Phuc Ho, Phuc Truong Loc Nguyen, Ren\'e Richard, Hung Cao  
**Category**: cs.AI  
**Published**: 2026-08-07  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.05391v1  

#### Abstract
Care plan coordination demands synthesizing heterogeneous clinical, functional, and psychosocial information across multiple professional disciplines, where monolithic LLM pipelines cannot perform in a transparent or safe manner. We present CANOE (Contestable Argumentative Network-of-Experts), a mul...

---

### 19. [Refining Over Resampling: Test-Time Self-Correction for LLM Reasoning](https://arxiv.org/abs/2608.05643)

**Authors**: Ahsan Bilal, Muhammad Ahmed Mohsin, Muhammad Umer, Lena Trigg, Ali Subhan, Muhammad Ali, Dean F. Hougen  
**Category**: cs.AI  
**Published**: 2026-08-07  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.05643v1  

#### Abstract
Test-time scaling improves LLM reasoning by using additional inference compute, but wider sampling alone can suffer from diminishing returns: new rollouts often repeat existing answer patterns instead of adding useful reasoning diversity. Verifier-based selection offers an alternative, but its perfo...

---

### 20. [DreamGuard: Efficient Runtime Guardrail for LLM Agents via Risk-Aware World Model](https://arxiv.org/abs/2608.05695)

**Authors**: Wenhao Lin, Chenyu Yu, Xingwei Lin, Sicong Cao, Xiang Chen, Lei Xue, Le Yu, Letian Sha, Chunming Wu  
**Category**: cs.AI  
**Published**: 2026-08-07  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.05695v1  

#### Abstract
As large language model (LLM) agents increasingly invoke external tools and interact with real-world systems, unsafe actions may cause irreversible consequences on external states, user data, and downstream services. Recent runtime guardrails mitigate such risks by checking proposed actions before e...

---

### 21. [Hybrid-Adaptive Thread Tuning to Mitigate Simulation Execution Bottlenecks in High-Performance Reinforcement Learning Inference](https://arxiv.org/abs/2608.06025)

**Authors**: Jiming Su, Hantao Hua, Lujia Yin, Yiping Yao, Feng Zhu  
**Category**: cs.LG  
**Published**: 2026-08-07  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.06025v1  

#### Abstract
In simulation-in-the-loop decision-making systems, reinforcement learning (RL) inference is often constrained by simulator-side execution overhead, where workloads are highly dynamic and sensitive to runtime thread configurations. Existing multithreaded strategies struggle to match thread resources ...

---

### 22. [MetaboLLM: a metabolomics-specialized large language model for biochemical knowledge integration and predictive metabolite graph construction](https://arxiv.org/abs/2608.06253)

**Authors**: Dohyun Ku, Min Gu Kwak, Francisco J. Pasquel, Jing Li  
**Category**: cs.LG  
**Published**: 2026-08-07  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.06253v1  

#### Abstract
Metabolomics knowledge is distributed across heterogeneous resources and remains difficult to translate into predictive representations. We developed MetaboLLM, a metabolomics-specialized large language model adapted through continual pretraining, supervised fine-tuning, and structured retrieval, to...

---

### 23. [Surv-IPTB: An Attention-Based Model for Estimating Individual Probability of Treatment Benefit with Survival Data](https://arxiv.org/abs/2608.06288)

**Authors**: Lev V. Utkin, Stanislav K. Kogan, Andrei V. Konstantinov  
**Category**: cs.LG  
**Published**: 2026-08-07  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.06288v1  

#### Abstract
This work presents a novel attention-based framework for estimating the Individual Probability of Treatment Benefit (IPTB) in survival analysis contexts. The proposed model, called Surv-IPTB, directly quantifies the probability that a specific patient will experience extended survival time under tre...

---

### 24. [LUNAR: Benchmarking Personalized Large Language Models on UNiversal User BehAvioR Logs](https://arxiv.org/abs/2608.05246)

**Authors**: Jiahao Zhang, Yongzhi Tong, Zelin Fu, Pengde Zhao, Yanmei Jiang, Jiang Feng, Min Yang  
**Category**: cs.AI  
**Published**: 2026-08-07  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.05246v1  

#### Abstract
Existing personalized LLM benchmarks primarily rely on textual personas or isolated behavioral signals, providing limited evaluation of cross-domain behavioral personalization, where responses must be grounded in heterogeneous daily-life activities. To address this gap, we introduce LUNAR, the first...

---

### 25. [DoctorAgents: an agentic framework to iteratively refine AutoML pipeline for small clinical temporal data](https://arxiv.org/abs/2608.05375)

**Authors**: Ruilin Wang, Bo-Hong Wang, Elizabeth Kourbatski, Jun Bai, Hegang Chen, Ziyang Song, Gilles Boire, Marie Hudson, Yue Li  
**Category**: cs.AI  
**Published**: 2026-08-07  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.05375v1  

#### Abstract
Clinical machine learning (ML) has the potential to support high-stakes medical decision-making, but reliable deployment is often constrained by scarce, heterogeneous, and temporal complexity. Developing effective ML pipelines for such data remains time-consuming and error-prone, while existing auto...

---

### 26. [Cautious Context Steering for Language Model Personalization](https://arxiv.org/abs/2608.05813)

**Authors**: Gihoon Kim, Jeyoung Lee, Suhan Woo, Sekwon Oh, Minsu Jeon, Hyounsoo Han, Euntai Kim  
**Category**: cs.AI  
**Published**: 2026-08-07  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.05813v1  

#### Abstract
Personalizing language models (LMs) to individual user preferences is essential for aligning responses with diverse goals and backgrounds. Existing methods typically train a separate adapter for each user or learn a reward model whose scores depend on the user. Despite explicitly optimizing for each...

---

### 27. [GSBF: Gaussian Splatting for Environment-Aware Beamforming](https://arxiv.org/abs/2608.05896)

**Authors**: Yijie Bian, Wei Guo, Zixin Wang, Shenghui Song, Jun Zhang, Khaled B. Letaief  
**Category**: cs.AI  
**Published**: 2026-08-07  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.05896v1  

#### Abstract
Beamforming plays a key role in multiple-input-multiple-output (MIMO) communication systems. However, conventional beamforming design normally requires accurate instantaneous channel state information (CSI) and iterative optimization, which incur substantial pilot overhead and computational complexi...

---

### 28. [Temporal Bridges for Spatial Resolution: Enhancing Climate Data Super-Resolution with Bidirectional Alignment](https://arxiv.org/abs/2608.05981)

**Authors**: Yichen Zhang, Yixiong Xiao, Congxi Xiao, Jingbo Zhou  
**Category**: cs.AI  
**Published**: 2026-08-07  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.05981v1  

#### Abstract
High-resolution climate data is crucial for meteorological predictions and for informing decision support across diverse domains. However, the acquisition of such high-resolution climate information is often prohibitively costly, necessitating the development of data-driven meteorological prediction...

---

### 29. [HERALD: Counterfactual Audits and Minimal Repairs for Proof-of-Retrieval Rewards](https://arxiv.org/abs/2608.06012)

**Authors**: Zhuowen Liu, Bohan Cui, YinShang Guo, Yuting Wang, Hao Li  
**Category**: cs.AI  
**Published**: 2026-08-07  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.06012v1  

#### Abstract
Search-agent rewards mix answer quality, citation grounding, tool cost, and anti-hacking terms; a high score therefore need not imply that cited evidence was retrieved, and added penalties can cancel. We introduce HERALD, an offline audit that applies exact same-question interventions, separates can...

---

### 30. [ECHO: A Locally-Deployable Agentic Health Assistant with Temporal Memory, Safety Guardrails, and Speech Assessment](https://arxiv.org/abs/2608.06110)

**Authors**: Abdulkadir K\"ul\c{c}e, Alihan Esen, Ca\u{g}la Fikir, Berke Kurt, Kuzey Arar, G\"okhan Ercan, Faik Boray Tek  
**Category**: cs.AI  
**Published**: 2026-08-07  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.06110v1  

#### Abstract
This paper presents ECHO (Enhanced Care \& Health Observer), a locally-deployable conversational health assistant for long-term chronic care management. ECHO integrates three complementary software modules developed under shared supervision as a unified system. The core module is an agentic chatbot ...

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

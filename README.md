# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-10-13 12:59:53 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Contextual Correction of Parallel Draftsfor Speculative Decoding](https://arxiv.org/abs/2610.12570)

**Authors**: Roy Eisenstadt, Harel Lidar, Tal Gonen, Omer Luxembourg, Itamar Zimerman, Lior Wolf, Eli David, Ilan Naiman  
**Category**: cs.CL  
**Published**: 2026-10-13  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2610.12570v1  

#### Abstract
Speculative decoding accelerates large language model inference by using a lightweight drafter to propose multiple tokens that are subsequently verified in parallel by the target model. Recent diffusion-based drafters make this paradigm particularly attractive: deep attention-based neural backbones ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Contextual Correction of Parallel Drafts for Speculative Decoding 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **Speculative Decoding**（SD）框架中，轻量级的 **drafter** 模型并行生成多个候选 token，随后由目标模型（verifier）进行验证。虽然基于 **diffusion** 的并行 drafter（如 DFlash）能显著降低生成延迟，但由于其在预测每个位置时无法访问前序已确定的 token，导致 **intra-block dependencies**（块内依赖性）缺失，从而降低了 draft token 的接受率。

现有方法（如 DSpark、Domino）通过引入 **sequential correction 模块** 来恢复这些依赖关系，但这会将额外的串行计算引入关键路径，增加了每轮循环的延迟，尤其在高并发场景下得不偿失。

### 提出了什么新方法或新思路
本文提出 **DCOCO**（DFlash Contextual Corrector），一种轻量级的上下文校正机制，用于修复并行 diffusion drafter 生成的 draft blocks 中的不一致性，同时保持完全并行的计算特性。

DCOCO 的核心思想是：
- 不直接修改 drafter 的输出，而是保留每个位置的 top-K 候选 token；
- 利用 drafter 单次前向传播中的隐藏状态（hidden states）、logits 和 token embeddings，在一个 **共享的 MLP scorer** 中并行计算所有相邻位置间的候选转移得分（transition scores）；
- 最后采用 **prefix-first greedy selection rule**，从左到右依次选择最优转移路径，形成最终的 draft proposal。

该方法的关键在于：
- 所有评分计算是 **batched 并行完成** 的，不增加 drafter 的前向 pass；
- 只有最后的选择过程是顺序的，且成本极低（O(BK)）；
- 无需训练额外的序列模块，避免了串行瓶颈。

### 相比现有方法的优势
- **更高的吞吐量（Throughput）**：在多种配置下均取得最高 end-to-end 解码速度，相比 vanilla decoding 加速高达 **6.6×**。
- **更强的鲁棒性**：在高并发（concurrent serving）场景下仍保持优势，而 DSpark 和 Domino 在负载增加时因串行开销过大反而低于未校正的 DFlash。
- **更高效的校正机制**：仅利用已有信息进行重打分与重选择，而非重新生成或迭代修正，实现了“低成本高回报”的校正。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验覆盖 **10 个 benchmark 数据集**，涵盖三大领域：
- **数学（Math）**: `MATH-500`, `AIME-25`, `GSM8K`
- **代码（Code）**: `HumanEval`, `MBPP`, `LiveCodeBench`
- **通用任务与对话（General）**: `MT-Bench`, `Arena-Hard-v2`, `Alpaca`, `SpeedBench`

所有数据集均使用固定子集进行评估，确保公平比较。

### 实验设置和评估指标
#### 模型配置
- **Verifier backbones（验证器主干）**:  
  `Qwen3-4B`, `Qwen3-8B`, `Qwen3-Coder-30B-A3B`, `Llama-3.1-8B`
- **Drafter block size**: 16
- **Candidate retention K**: 16
- **Hardware**: 单张 NVIDIA B200 GPU
- **Framework**: vLLM

#### 解码模式
- **Greedy decoding** (`T=0`)
- **Stochastic decoding** (`T=1`)

#### 评估指标
- **Primary Metric**:  
  - **End-to-end decoding throughput**（TPS, tokens per second）
  - **Speedup over vanilla autoregressive decoding**
- **Mechanism Metric**:  
  - **Mean accepted length** $ \mathbb{E}[A(\cdot)] + 1 $：衡量 proposal 质量
- **Concurrency scaling**: 测试 1–32 并发请求下的系统吞吐表现

### 基线方法对比
- **Vanilla decoding**: 无 speculative decoding 的标准自回归解码
- **DFlash**: 未经校正的并行 diffusion drafter（baseline）
- **DSpark**: 引入 semi-autoregressive correction 的串行增强方法
- **Domino**: 使用因果编码器进行 post-hoc 校正的方法

所有基线方法均在相同训练语料、硬件和服务栈上重新训练与部署，保证公平性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 配置 | 最高平均加速比（vs. Vanilla） |
|------|-------------------------------|
| Greedy decoding | 最高达 **6.6×** |
| Stochastic decoding | 最高达 **2.95×**（见 Table 2） |

在 `Qwen3-8B` 上的具体表现（Table 1）：
- **Greedy (T=0)**: DCOCO 达到 **959.7 TPS**，比最强基线（Domino）提升 **14.3%**
- **Stochastic (T=1)**: DCOCO 达到 **762.2 TPS**，比最强基线提升 **13.6%**

### 与基线方法的对比结果
#### 综合性能（Table 2）
DCOCO 在 **全部 12 种配置**（4 个 verifier × 是否开启 thinking × 两种温度）中均取得最高的平均加速比，领先第二名 **8–19%**。

例如：
- `Qwen3-8B`, `T=0`, 无 thinking：**3.62×** vs Domino 的 3.16×
- `Llama-3.1-8B`, `T=0`：**3.75×** vs Domino 的 3.24×

#### 高并发场景表现（Figure 3 & Table 19）
随着并发数从 1 增加到 32：
- **DCOCO 始终保持领先**，在 `Qwen3-8B` 上相对最强基线的增益从 **14.3% 提升至 19.2%（T=0）**
- **DSpark 和 Domino 性能下降明显**，在高负载下甚至 **低于未校正的 DFlash**，因其串行校正模块成为性能瓶颈

#### 接受长度与吞吐关系分析（Figure 4）
- DCOCO 同时提升了 **accepted length** 和 **cycle efficiency**
- 相比之下，DSpark 和 Domino 的接受长度虽有提升，但被高昂的校正延迟抵消，导致整体吞吐下降

### 消融实验结果（Table 5 & 6）
对 DCOCO 内部组件进行消融：
| 消融项 | 对比对象 | TPS 下降（T=0） | Accepted Length 下降 |
|--------|----------|------------------|------------------------|
| `-pair state` | 完整 DCOCO | 6.3% | 7.3% |
| `-cand state` | 完整 DCOCO | 4.6% | 4.8% |
| `Frozen backbone` | Joint training | 15.9% | 16.5% |

结论：
- **pair state** 和 **candidate representation** 都对性能至关重要
- **联合训练**（joint training）带来显著收益，说明 drafter 与 corrector 需协同优化

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Top-K 候选中蕴含大量正确 token**：尽管 drafter 的 top-1 准确率随位置加深迅速下降，但 top-16 候选池中仍以较高概率包含 verifier 实际选择的 token（Figure 1, Table 18）。这为 “re-selection” 提供了可行性基础。
   
2. **并行上下文评分可行且高效**：通过复用 drafter 的中间表示，并行计算所有候选转移得分，可在几乎零额外延迟的前提下实现高质量重排序。

3. **prefix-first selection 更符合 speculative verification 的结构**：由于验证过程一旦失败即停止，因此应优先保障前缀质量。全局路径搜索（global path search）在此场景下并非最优。

4. **校正机制必须轻量化**：任何引入串行处理的校正方法在高并发下都会遭遇可扩展性瓶颈；只有像 DCOCO 这样将主要计算保持并行的方法才能持续受益。

### 方法的局限性
- 当前设计依赖于特定的 drafter 架构（如 DFlash）提供丰富的中间表示；
- 正确性依赖于 drafter 至少在 top-K 中包含正确 token，若 drafter 质量过差则无法补救；
- 仅适用于 block-wise speculative decoding 场景，难以推广至 tree 或 graph 结构的 speculation。

### 未来工作方向
- 将 DCOCO 思路扩展至其他类型的 drafter（如 Medusa、Hydra）；
- 探索动态调整 K 值或 block size 的策略以进一步优化效率；
- 研究如何将类似 contextual correction 应用于 **extended thinking / reasoning chains** 中的长程依赖建模；
- 探索更复杂的 selection rule（如 beam search），在可控延迟下探索性能边界。

---

> ✅ **总结一句话**：  
> DCOCO 证明了——**通过并行地对已有候选进行上下文重打分与贪婪前缀选择，可以在不牺牲延迟优势的前提下有效恢复并行 drafter 的块内依赖性，从而在各种 decoding regime 和 serving load 下实现 SOTA 的 speculative decoding 性能。**

</details>

---

### 2. [Adaptive DHiSparse: Speculative Decoding over a Load-Adaptive KV Hierarchy](https://arxiv.org/abs/2610.13708)

**Authors**: Naji Alazhar, Wassel Alazhar  
**Category**: cs.AI  
**Published**: 2026-10-13  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.13708v1  

#### Abstract
Long-context serving runs out of KV memory before the hardware reaches its useful throughput. HiSparse stores history in host memory and fetches only the selected entries, but it uses one cache budget at every load and does not handle draft-model speculative decoding such as DSpark and DFlash2. We p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Adaptive DHiSparse: Speculative Decoding over a Load-Adaptive KV Hierarchy**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前大模型在长上下文（long-context）推理场景下面临两个核心瓶颈：
1. **KV Cache 内存瓶颈**：传统方法将完整的 KV Cache 存储在 GPU 高带宽内存（HBM）中，导致在高并发或长上下文下迅速耗尽显存，限制了可服务的请求数量。
2. **缺乏对 Speculative Decoding 的支持**：现有的分层 KV 缓存系统（如 HiSparse）不支持 draft-model 类型的 speculative decoding（如 DSpark、DFlash2），无法利用 draft 模型加速推理。

此外，固定大小的缓存策略（如 HiSparse）在负载变化时表现不佳：低负载时浪费 HBM 容量，高负载时又无法动态释放内存以容纳更多请求。

---

### **提出的新方法与创新思路**
论文提出了 **Adaptive DHiSparse**，一个支持 speculative decoding 的、负载自适应的分层稀疏 KV 缓存系统，其核心创新包括：

#### **(1) 负载感知的 KV 居住性管理（Load-Adaptive Residency）**
- **保护核心工作集**：保留一个固定的“热缓存”（hot cache, B 行）用于保证基本的命中率和快速恢复。
- **可选历史驻留（Optional History）**：当 HBM 有空闲容量时，将更多历史 KV 行保留在 GPU 上，减少从 Host DRAM 的读取；当负载上升时，主动回收这些页面供新请求使用。
- **resident-first admission**：新请求的历史优先驻留在 GPU，仅在内存压力下才触发备份到主机。

> ✅ 实现了 **运行时动态调整**，而非部署期静态配置。

#### **(2) 支持 Speculative Decoding 的统一执行路径**
- 在分层 KV 架构上完整支持 **DSpark 和 DFlash2** 的 speculative decoding 流程：
  - 多查询共享缺失行检索（fused miss resolution）
  - draft KV 使用窗口有界环形缓冲区（window-bounded ring buffer）
  - 验证阶段的临时状态复用 staging arena
- 所有 speculative 查询共享历史读取，避免重复 fetch。

#### **(3) 受控的 speculative 存储开销**
- **Draft KV Ring Buffer**：大小仅取决于 draft 模型可见窗口 $W$ 和最大验证宽度 $K_{\text{max}}$，与上下文长度无关。
- **可复用的验证 staging arena**：多个 layer group 共享 staging 区域，显著降低临时存储需求。

---

### **相比现有方法的优势**
| 维度 | HiSparse | Adaptive DHiSparse |
|------|--------|------------------|
| KV 层级支持 | ✅ 分层缓存 | ✅ 更优的负载适配 |
| 可变缓存大小 | ❌ 固定预算 | ✅ 动态增减 optional history |
| 支持 speculative decoding | ❌ 不支持 | ✅ 完整支持 DSpark / DFlash2 |
| 内存效率 | 中等 | 显著提升，尤其在高并发 |
| 性能扩展性 | 受限于固定缓存 | 随负载弹性伸缩 |

> 🔥 **核心优势**：首次实现 **speculative decoding + hierarchical KV + load-adaptive residency** 的三位一体，兼顾高吞吐、低延迟与资源利用率。

---

## **2. 核心实验方法和设置**

### **使用的模型与硬件平台**
- **模型**：
  - `GLM-5.3`：100K 输入上下文，输出上限 2,048 tokens
  - `DeepSeek-V4.1-Flash`：500K 输入上下文，输出上限 2,048 tokens
- **硬件**：
  - 8× B300 GPU（每卡 269 GiB HBM）
  - 主机内存 4.3 TB via PCIe

### **实验设置**
- **工作负载**：闭源循环并发基准测试（closed-loop concurrency benchmark）
- **变量控制**：
  - 固定输入长度、输出上限
  - 使用预准备的 prefix（prepared prefixes）
  - 控制 offered concurrency（C）从 4 到 512
- **缓存参数匹配**：
  - `k = 2048`, `B = 4096`（GLM）
  - `k = 512`, `B = 8192 / 4096`（DeepSeek）

### **评估指标**
| 指标 | 含义 |
|------|------|
| **Output Throughput (tok/s/GPU)** | 每个 decode GPU 每秒生成的 token 数 |
| **Mean TTFT (Time to First Token)** | 平均首 token 延迟 |
| **Mean TPOT (Time Per Output Token)** | 平均每个输出 token 的处理时间 |
| **Post-wave Latency** | 排除前 C 个请求后的稳定态延迟 |
| **Active Sessions** | 实际同时解码的请求数量 |

### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **Resident Serving** | 完整 KV 驻留 GPU HBM |
| **HiSparse** | 固定缓存大小的分层稀疏 KV |
| **Adaptive HiSparse** | 无 speculative 的自适应版本 |
| **Adaptive DHiSparse** | 本文方法（含 speculative） |

所有方法基于 **SGLang** 框架实现，计算内核保持一致，确保公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

#### **GLM-5.3 @ 100K context, C128**
| 方法 | Output (tok/s/GPU) | vs HiSparse | Mean TTFT (s) | Mean TPOT (ms) |
|------|--------------------|-------------|---------------|----------------|
| HiSparse | 548 | — | 10.68 | 50.3 |
| Adaptive HiSparse | 622 | **1.14×** | 6.13 | 46.6 |
| Adaptive DHiSparse | **793** | **1.45×** | **5.13** | **35.9** |

> 📈 在相同延迟约束下（p99 TTFT ≤ 10s, TPOT ≤ 40ms）：
> - HiSparse 最多支持 **C32**
> - Adaptive DHiSparse 支持 **C128**
> - → **并发能力提升 4×，总输出提升 3.7×**

#### **DeepSeek-V4.1-Flash @ 500K context, C512**
| 方法 | Output (tok/s/GPU) | vs Resident |
|------|--------------------|-------------|
| Resident + DSpark | 2,730 | — |
| Adaptive DHiSparse + DSpark | **5,054** | **1.85×** |

> 🚀 即使在极端长上下文（500K）和超高并发（512）下仍实现近两倍吞吐提升。

---

### **消融实验结果**

#### **(1) 可选驻留（Optional Residency）的影响**
- 在低负载（C16）时：
  - Host-fetch 比例从 **14.2% → 1.4%**
  - per-session 速度提升 **3.8%**
- 在负载下降后恢复：
  - Host-fetch 下降 **5.8×**
  - 会话速度提升 **13.1%**
- ⚠️ 缺点：负载突增时需 53 秒完成页面回收（vs 3s 无驻留）

> ✅ 证明 optional residency 在非峰值时段有效节省 IO 开销。

#### **(2) Speculative Block Size 影响**
- 在 C64：block=6 最优（比 block=4 输出高 ~18%）
- 在 C128/C256：block=4 最优，更大 block 导致排队延迟增加，收益被抵消
- 结论：应根据负载动态选择 block size

#### **(3) Hot Cache 大小影响（B）**
- **GLM-5.3**：存在明显 trade-off
  - B↑ → miss↓ → step time↓
  - 但占用内存↑ → 可服务 session↓
  - 最优值出现在 **B=2k**（池容量为 2k 时）
- **DeepSeek-V4.1-Flash**：output 对 B 不敏感
  - 原因：更少的 selection layers（8 vs 21）、更窄的 row width（288B vs ~2.4KB）
  - 每 token host fetch 量低约 **80×**

> 💡 模型结构设计直接影响 offloading 效果。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Adaptive DHiSparse 显著提升长上下文服务吞吐**
   - 在 100K 上达到 **1.45× HiSparse**, **1.27× 自适应非 speculative**
   - 在 500K 上实现 **1.85× resident + DSpark**

2. **负载自适应机制有效平衡内存使用**
   - 低负载时利用空闲 HBM 减少 host fetch（降幅达 83–90%）
   - 高负载时快速释放内存保障 admission capacity

3. **Speculative Decoding 与 Hierarchical KV 可协同优化**
   - 共享 missing-row retrieval 减少重复 fetch
   - draft ring + reusable staging arena 控制额外内存开销（仅 +0.060 GiB/session）

4. **模型架构深刻影响 offloading 成本**
   - DeepSeek-V4.1-Flash 因 shared KV source 和 smaller k，host fetch 成本远低于 GLM-5.3
   - 未来模型设计应考虑 “KV friendliness”

---

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **Host Fetch 带宽依赖** | 性能受限于 PCIe 带宽，若 host DRAM 延迟高则收益下降 |
| **回收延迟** | 页面回收过程耗时数十秒，可能影响突发流量响应 |
| **未支持在线调整 block size** | 当前 speculative block size 为静态配置，未根据实时负载动态选择 |
| **仅限特定 speculative 方法** | 目前仅支持 DSpark / DFlash2，未涵盖 Medusa 或 EAGLE 等 |

---

### **未来工作方向**
1. **动态调整 speculative 参数**
   - 根据当前负载、内存可用性、miss rate 实时调节 block size
2. **更快的页面回收机制**
   - 引入异步迁移、预测性预淘汰等技术缩短 reclamation 时间
3. **扩展至其他 speculative 范式**
   - 支持 Medusa、EAGLE 系列等基于 draft tree 的 speculative 方法
4. **跨节点分布式适配**
   - 将 adaptive residency 扩展到 multi-node 场景，协调全局内存分配
5. **结合 Prefetch 技术**
   - 利用访问模式预测提前加载 likely-to-be-used KV 行

---

> ✅ **总结一句话**：  
> **Adaptive DHiSparse 成功将 speculative decoding 引入 hierarchical KV 架构，并通过 load-adaptive residency 实现了内存使用与性能之间的最优权衡，在长上下文高并发场景下实现了高达 1.85× 的吞吐提升。**

</details>

---

### 3. [Learning More by Training Less: Uncertainty-Guided Progressive Contraction for Efficient Reasoning Post-Training](https://arxiv.org/abs/2610.12933)

**Authors**: Suorong Yang, Jiaxin Wang, Jiaming Su, Chentao Yue, Furao Shen, Peng Ye, Soujanya Poria  
**Category**: cs.AI  
**Published**: 2026-10-13  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.12933v1  

#### Abstract
Large language models have achieved remarkable advances in complex reasoning. However, scaling reasoning post-training with ever-larger corpora can substantially increase training cost without proportional performance gains. In this study, we therefore ask: how far can reasoning supervision be compr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Learning More by Training Less: Uncertainty-Guided Progressive Contraction for Efficient Reasoning Post-Training*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前大型语言模型（LLM）在复杂推理任务上的后训练（post-training）通常依赖于大规模的推理轨迹数据（如 Chain-of-Thought, CoT），导致训练成本高昂且边际收益递减。本文提出一个核心问题：**能否在不牺牲推理能力的前提下，大幅压缩推理监督信号的数据量？**

传统方法要么简单增加数据量，要么基于正确性、难度或质量进行静态过滤，未能动态结合教师生成行为与学生学习状态来识别高价值监督路径。

---

### 提出的新方法与创新思路
作者提出 **Uncertainty-Guided Progressive Contraction Framework**（不确定性引导的渐进收缩框架），其核心思想是通过两阶段渐进式收缩，将优化集中在最具学习价值的推理路径上：

#### （1）**教师侧不确定性建模（Teacher-side Uncertainty Profiling）**
- 构建 **Monte Carlo Reasoning Tree (MCRT)**：对每个中间推理状态，采样多个延续路径，统计最终答案的正确率和分歧程度。
- 定义 **过程级不确定性（process-level uncertainty）**：使用归一化伯努利方差 $ U_{i,t} = 4\hat{p}_{i,t}(1-\hat{p}_{i,t}) $ 来衡量从某状态出发的不同延续是否导向不同结果。
- 聚合为轨迹级不确定性得分 $ U_i $，保留高不确定性的 CoT 组成紧凑候选池 $ \mathcal{P}_p $。

> ✅ 创新点：首次将“不确定性”作为推理能力边界的代理信号，而非仅依赖终端正确性或置信度。

#### （2）**学生感知的动态路由（Learner-aware Routing）**
- 在 SFT 过程中引入一次探测轮次（probing epoch），计算每个样本对学生当前状态的响应损失（response loss）。
- 动态选择损失最高的子集 $ \mathcal{H}_M $ 进行后续优化，实现“越难学、越优先”的自适应训练调度。

> ✅ 创新点：将监督构建与下游优化分配联合设计，形成“先筛选高信息量路径 → 再聚焦学生未掌握路径”的闭环。

---

### 相比现有方法的优势
| 维度 | 本方法优势 |
|------|-----------|
| **效率** | 仅需 **0.22% ~ 0.74% 的优化预算** 即可达到甚至超越全量 SFT 性能，训练时间减少 **>450×** |
| **通用性** | 支持跨架构（Qwen/Gemma/Llama/Mistral）、黑盒教师（API-only）、弱教强学（weak-to-strong）场景 |
| **有效性** | 不仅加速训练，还能提升最终推理性能（capability gain） |
| **实用性** | 完全基于可观测行为（文本输入输出），无需访问模型内部参数或梯度 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- 主要训练数据：**DAPO-17K**（开源的大规模数学推理强化学习数据集）
- 黑盒教师实验：**Nemotron-Math** 子集（前1000个样本）
- 评估涵盖三大类推理任务：
  - **竞赛数学**：AIME24/25/26, AMC, HMMT25/26
  - **通用数学**：MATH-500, GSM8K
  - **科学推理**：GPQA-D

---

### 实验设置
- **教师模型**：
  - 开源：Qwen3-8B, Qwen3-1.7B
  - 黑盒 API：DeepSeek-V4-Pro, Kimi-K2.6
- **学生模型**：
  - Qwen3.5-4B / 9B
  - Gemma-4-E4B-IT, Llama-3.2-3B-Instruct, Mistral-7B-Instruct
  - 强学生：Qwen3.5-14B（接受弱教师指导）
- **训练方式**：LoRA-based SFT，控制总优化步数和样本暴露次数
- **评估指标**：
  - 准确率（Accuracy）在各基准测试集
  - 训练预算（Training Budget）：相对全量 SFT 的样本暴露比例
  - 实际开销：GPU 小时（wall-clock time）

---

### 基线方法对比
- **Full SFT**：在全部验证通过的 CoT 上进行标准三轮 SFT
- **Vanilla Distillation**：使用相同数量的标准 CoT 进行蒸馏
- **Random Selection**：随机选取相同比例的高质量 CoT
- **Top-k Confidence**：选择教师最自信的 CoT（反向验证其低效性）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| 方法 | 预算 | AIME26 (4B) | HMMT25 (4B) | GPQA-D (4B) |
|------|------|-------------|--------------|--------------|
| Full SFT | 100% | 66.7 | 40.0 | 75.8 |
| Ours (p=10%) | **0.74%** | **83.3 (+16.6)** | **76.7 (+36.7)** | **77.3 (+1.5)** |
| Ours (p=5%) | **0.22%** | 80.0 | 76.7 | 77.8 |

> ⚡️ 结果显示：**仅用 0.22% 的训练预算，性能全面超越 Full SFT**，尤其在竞赛数学上提升显著。

---

### 与其他方法对比（Table 5）
在 **1% 训练预算下**，相比 Vanilla Distillation：

| 学生 | 方法 | 平均准确率提升 |
|------|------|----------------|
| Qwen3.5-4B | Ours vs Vanilla | **+3.6%** |
| Qwen3.5-9B | Ours vs Vanilla | **+4.0%** |

说明：**不确定性引导的选择比随机或固定策略更有效**。

---

### 消融实验结果（Table 6）

| 组件组合 | AIME24 ↑ | HMMT26 ↑ | 效果分析 |
|--------|---------|----------|--------|
| None | 83.3 | 54.5 | 基线 |
| Only Teacher Uncertainty | 80.0 | 57.6 | 可能选到学生已掌握的难点 |
| Only Learner Routing | 83.3 | 63.6 | 忽略路径本身的信息密度 |
| **Both (Ours)** | **86.7** | **78.7** | ✅ 显著增益，证明两个阶段互补 |

> 🔍 发现：**教师不确定性 + 学生感知路由 是协同增效的关键**。

---

### 不确定性 vs 随机选择（Table 4）
在相同预算下，不确定性选择始终优于随机抽样：
- 在 5% 池大小下，平均性能高出 **+4.4%**
- 表明：“正确的 CoT” ≠ “有教学价值的 CoT”，**过程不确定性才是关键区分因素**

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **推理能力边界可通过过程不确定性刻画**：即使正确或高置信的 CoT，也可能隐藏脆弱推理分支（见 Fig. 1）。
2. ✅ **少量高不确定性路径蕴含巨大学习价值**：仅需极少样本即可完成高效知识迁移。
3. ✅ **学生状态应参与优化分配**：静态过滤不足以最大化效率，必须动态适配学习进程。
4. ✅ **方法具备极强泛化性**：
   - 跨模型家族（Qwen/Gemma/Llama/Mistral）
   - 跨教师模式（开源 / API-only 黑盒）
   - 支持 **weak-to-strong transfer**（1.7B 教师提升 14B 学生）

---

### 方法的局限性
- **依赖高质量教师生成**：若教师本身推理不稳定，MCRT 可能无法提取可靠不确定性信号。
- **MCRT 构造有一定开销**：虽然远小于训练节省的成本，但仍需额外采样与验证。
- **目前聚焦于数学与科学推理**：在开放域对话或多模态任务中的扩展尚待验证。

---

### 未来工作方向
- 将不确定性信号用于 **在线课程设计（curriculum learning）**
- 探索 **完全无教师的自我不确定性挖掘机制**
- 扩展至 **多跳检索、规划、工具调用等复杂推理链任务**
- 结合 RLHF 或 DPO 进一步优化对齐效率

---

## 总结
该论文提出了一个极具实用价值的范式转变：**从“越多越好”转向“更聪明地使用”推理数据**。通过引入 **过程级不确定性** 和 **渐进式收缩机制**，实现了 **极致的数据效率与性能提升并存**，为未来低成本、高可及性的 LLM 推理能力训练提供了新路径。

</details>

---

### 4. [Masked Diffusion Language Agents for Tool-Integrated Chemical Reasoning](https://arxiv.org/abs/2610.13330)

**Authors**: Mengdi Liu, Chenghao Jia, Hong Chang, Shiguang Shan  
**Category**: cs.AI  
**Published**: 2026-10-13  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.13330v1  

#### Abstract
Designing tool-integrated reasoning agent systems for complex chemical tasks remains a fundamental challenge. Although recent chemical agents have demonstrated promising performance through the integration of external tools, they are still built on the autoregressive paradigm, which is inherently co...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Masked Diffusion Language Agents for Tool-Integrated Chemical Reasoning

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

传统基于 **autoregressive (AR)** 范式的化学推理智能体（chemistry agents）在处理复杂化学任务时面临以下根本性挑战：

- **单向生成限制**：AR模型采用从左到右的生成方式，无法进行双向上下文建模，难以理解具有复杂依赖关系的化学表示（如SMILES字符串）。
- **局部决策偏差**：工具调用决策是逐步串行做出的，缺乏对整个推理轨迹的全局规划能力，容易导致工具选择错误或执行不稳定。
- **多源证据整合困难**：复杂化学任务需要综合来自多个工具返回的结果，而AR模型在长序列中易丢失或扭曲中间信息。

这些问题使得现有化学智能体在**工具集成推理**（tool-integrated chemical reasoning）任务中表现受限。

---

### **提出了什么新方法或新思路**

本文提出 **CHEMDIFFAGENT**，这是首个将 **masked diffusion language model (dLLM)** 应用于工具集成化学推理的智能体框架。其核心思想是：

- 将化学推理过程重新定义为一个**迭代去噪**（iterative denoising）过程，在交错的“思考-工具调用”轨迹上进行联合优化。
- 采用 **Block Diffusion** 架构，允许在每个块内以任意顺序解码，实现更灵活、协调的决策。
- 设计了专门针对智能体行为的训练目标：**Agentic ELBO**，仅监督智能体可控部分（如`think`, `tool_call`），忽略外部工具返回内容。

---

### **相比现有方法的优势**

| 维度 | Autoregressive Agents | CHEMDIFFAGENT |
|------|------------------------|---------------|
| 上下文建模 | 单向（causal） | 全局双向 |
| 工具决策 | 逐个承诺（left-to-right） | 联合优化（joint refinement） |
| 推理灵活性 | 固定顺序 | 任意顺序解码（flexible ordering） |
| 多证据整合 | 易失真、丢失 | 支持槽位级融合（slot-level fusion） |
| 错误修正 | 难以回溯 | 可在去噪过程中自我修正 |

> ✅ **核心优势**：通过扩散范式实现了**决策边界与知识边界的更好对齐**，即智能体能更准确地判断何时应使用工具、使用哪个工具，并基于完整上下文做出可靠决策。

---

## 2. 核心实验方法和设置

### **使用的数据集**

作者构建了一个全面的化学智能体评测基准，涵盖六类任务，分为两种场景：

#### **单轮（Single-turn）任务**
- Composition Inference（组成推断）
- Name Conversion（名称转换）
- Property Prediction（性质预测）
- Retrosynthesis（逆合成）
- Molecular Optimization（分子优化）
- Forward Synthesis（正向合成）

#### **多轮（Multi-turn）任务**
- 基于**化学工具依赖图**（chemistry tool dependency graph）构建任务流图。
- 通过随机游走采样子图，并由强教师模型生成需调用多个工具的任务。
- 分为三个难度等级：
  - **Easy**: 2次工具调用
  - **Medium**: 3–5次
  - **Hard**: 6–7次

> 数据来源包括 SMolInstruct、ChemTool Bench、MT-Mol 和 T5Chem-style reaction data。

---

### **实验设置和评估指标**

#### **评估指标**

| 指标 | 定义 |
|------|------|
| **Answer Correctness** | 使用 GPT-4o 判断最终答案是否正确解决任务 |
| **Tool Selection Accuracy** | 模型调用的工具集合中，覆盖“黄金工具链”中预期工具的比例 |

> 对多轮任务还按调用轮数分组报告性能（Easy/Medium/Hard）。

#### **训练设置**
- **Backbone**: SDAR-8B（一种 Block Diffusion LLM）
- **两阶段后训练框架**：
  1. **Agentic SFT**（监督微调）：使用高质量教师轨迹进行块级去噪训练
  2. **Agentic VRPO**（方差缩减偏好优化）：对比成功与失败轨迹，提升长期决策质量
- 所有实验在 8×NVIDIA A100 GPU 上运行

---

### **基线方法对比**

| 类别 | 基线模型 |
|------|----------|
| **通用大模型 + ReAct** | GLM-5.1, Gemini 2.5 Flash, DeepSeek-V3.2, GPT-5.5, Claude Opus 4.7 |
| **专用化学智能体** | ChemCrow, S1-NexusAgent |
| **自回归基线** | Qwen3-8B + SFT |
| **未微调扩散模型** | SDAR-8B（无SFT） |

> 所有模型在相同工具接口、训练数据和推理预算下公平比较。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### **表1：单轮任务性能（Success Rate %）**

| Method | Comp. | Name Conv. | Prop. Pred. | Retro. | Mol. Opt. | Fwd. Synth. | **Total** |
|--------|-------|------------|-------------|--------|-----------|--------------|-----------|
| GPT-5.5 | 71.4 | 56.9 | 64.6 | 96.9 | 64.2 | 91.7 | 70.6 |
| Claude Opus 4.7 | 90.9 | 88.2 | 87.5 | 100.0 | 100.0 | 100.0 | **90.9** |
| Qwen3-8B + SFT | 76.6 | 53.9 | 87.5 | 78.5 | 74.5 | 63.3 | 69.1 |
| **SDAR-8B + SFT** | 91.3 | 78.4 | 94.8 | 96.9 | 75.5 | 98.3 | **89.0** |
| **SDAR-8B + SFT + RL** | **91.8** | **97.1** | **95.8** | 96.9 | **79.2** | 98.3 | **91.8** |

> 🔺 **相比同规模AR模型（Qwen3-8B+SFT）提升约20个百分点**

---

#### **表2：多轮任务性能（Success Rate %）**

| Method | Easy | Medium | Hard | **Total** |
|--------|------|--------|------|-----------|
| GPT-5.5 | 92.0 | 63.3 | 31.8 | 61.1 |
| Claude Opus 4.7 | 98.9 | 76.4 | 44.9 | 72.8 |
| Qwen3-8B + SFT | 74.5 | 42.8 | 20.5 | 44.0 |
| **SDAR-8B + SFT** | 75.9 | 66.8 | 43.0 | **62.3** |
| **SDAR-8B + SFT + RL** | **78.2** | **69.9** | **44.9** | **64.6** |

> 🔺 在Hard任务上，CHEMDIFFAGENT 表现接近 Claude Opus 4.7（44.9 vs 44.9），显著优于其他8B级模型。

---

### **消融实验与行为分析**

#### ✅ **工具调用效率更高**
- CHEMDIFFAGENT 的实际调用轮数最接近预期步数，避免冗余调用（如Gemini常过度调用）或不足调用（如GPT-5.5常漏调）。
- 输出长度更短，说明推理更简洁高效。

#### ✅ **工具选择准确性更高**
- 如图3所示，在多轮任务中，CHEMDIFFAGENT 的工具选择准确率持续领先，尤其在高复杂度任务中优势明显。

#### ✅ **支持灵活解码顺序**
- 可视化显示，模型先确定工具调用结构（如`<tool call>`标签、参数格式），再填充具体的SMILES等化学输入。
- 这种“先结构后细节”的模式减少了因提前承诺而导致的无效调用。

---

## 4. 关键结论和发现

### **主要发现**

1. **扩散范式优于自回归范式**  
   在工具集成化学推理任务中，masked diffusion 提供了更强的**全局协调能力**和**双向理解能力**，显著优于同规模的 AR 模型（+~20%）。

2. **8B模型媲美甚至超越超大规模前沿模型**  
   CHEMDIFFAGENT（8B）在多项任务上的表现可比肩甚至超过 GPT-5.5 和 Claude Opus 4.7，证明了**架构创新的重要性**。

3. **灵活解码路径带来实质收益**  
   理论分析表明，灵活的生成顺序可以在：
   - 子问题难度不均时选择最优路径（adaptive path advantage）
   - 决策风险不均时优先解决低风险项以降低整体风险（strict ordering advantage）

4. **支持鲁棒的自我修正机制**  
   扩散模型可在去噪过程中不断验证和修正中间结果，有效避免 AR 模型常见的“幻觉陷阱”和错误传播。

---

### **方法的局限性**

1. **模型与领域范围有限**
   - 当前评估限于 **8B 参数规模**、固定工具集和纯计算化学流程。
   - 尚未扩展至湿实验（wet-lab）环境或多学科科学推理。

2. **部署基础设施尚不成熟**
   - 相比成熟的 AR 生态系统，dLLM 缺乏高效的缓存、推测解码和推理优化支持。
   - 训练更大规模的 dLLM 在工程上更具挑战性。

3. **训练成本较高**
   - 扩散模型需多次去噪步骤，训练时间和资源消耗高于标准 SFT。

---

### **未来工作方向**

- 将 CHEMDIFFAGENT 扩展至更大规模模型（>10B）和更广泛的科学领域（如生物、材料）。
- 开发面向 dLLM 的高效推理引擎（如 KV cache 优化、并行去噪调度）。
- 探索与真实实验室自动化系统的集成，实现闭环科学发现。
- 构建更具挑战性的多模态、跨学科工具集成基准。

---

> 💡 **一句话总结**：  
> CHEMDIFFAGENT 通过引入 **masked diffusion** 范式，首次实现了在化学智能体中对“思考-工具调用”轨迹的**全局联合优化**，显著提升了复杂任务下的推理可靠性、工具协调性和多证据整合能力，为下一代科学智能体提供了新的技术路径。

</details>

---

### 5. [RAMP: Recurrent Adaptive Mixed Precision for Efficient Prefill in Recurrent-Depth Language Models](https://arxiv.org/abs/2610.13041)

**Authors**: Ke Wan, Chen Chen  
**Category**: cs.CL  
**Published**: 2026-10-13  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.13041v1  

#### Abstract
Recurrent-depth language models improve inference-time reasoning by repeatedly applying a shared network to refine internal representations, but the additional recurrent computation also increases inference cost. This overhead is particularly pronounced during long-context prefill, where the full in...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：RAMP: Recurrent Adaptive Mixed Precision for Efficient Prefill in Recurrent-Depth Language Models

---

## 1. 论文的主要贡献和创新点

### 解决的问题
Recurrent-depth language models 通过重复应用共享网络来提升推理能力，但在 **long-context prefill** 阶段，由于每次递归步骤都需要处理完整的输入序列，导致计算开销显著增加。传统的 **uniform weight-and-activation quantization** 虽可加速计算，但反复引入低精度运算会累积数值扰动，严重偏离 full-precision 轨迹。

本文系统性地研究了：  
> **如何在有限的精度预算下，最优分配量化操作在 recurrent depth 上的位置，以最小化最终表示的误差？**

### 提出的新方法：RAMP（Recurrent Adaptive Mixed Precision）
提出了一种无需训练的混合精度推理方法 —— **RAMP**，其核心思想是：
- 在 **早期 recurrent 步骤** 使用 **低精度（如 FP8）weight-and-activation 计算** 进行快速初步优化；
- 在 **末期 recurrent 步骤** 恢复 **高精度（如 BF16）计算**，以稳定并恢复接近 full-precision 的输出轨迹。

该策略基于一个关键理论洞察：  
> **早期引入的量化误差会被后续更多步的递归操作衰减，而晚期引入的误差则几乎无法被修正。**

### 相比现有方法的优势
| 方面 | RAMP 的优势 |
|------|-------------|
| **效率 vs 精度权衡** | 显著优于 uniform quantization 和 early high-precision 方案，在保持近 full-precision 质量的同时实现端到端加速 |
| **无需重训练** | 完全 post-training 方法，不修改模型结构或参数 |
| **理论支撑强** | 建立了量化误差传播的理论边界，并证明“早低晚高”是最优策略（Theorem 3.2） |
| **硬件友好** | 利用现代 GPU（如 H100）对 FP8 的原生支持，实际获得可观吞吐增益 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **WikiText**：主语言建模评估数据集，构建了嵌套上下文（nested contexts），长度从 128 到 4096 tokens，所有样本共享相同的尾部 64 token 预测目标，确保跨长度比较公平。
- **PG19**：独立的长文本泛化数据集，用于验证方法在更复杂文档上的鲁棒性，上下文长度为 512–4096 tokens。

### 实验设置
- **模型**：Huginn-0125（Geiping et al., 2026），默认 recurrent depth $ T = 32 $
- **RAMP 配置**：前 24 步使用 FP8 weight-and-activation quantization，后 8 步恢复 BF16
- **硬件平台**：NVIDIA H100，启用 native FP8 matrix multiplication
- **优化路径**：所有配置均使用相同的 optimized empty-cache prefill 路径，排除实现差异影响

### 评估指标
| 指标 | 描述 |
|------|------|
| **△NLL** | 相对于 BF16-32 的负对数似然变化（paired teacher-forced NLL over last 64 tokens） |
| **Hidden l2 ↓** | 最终隐状态相对于 BF16 的相对 l2 误差（越小越好） |
| **Logit l2 ↓** | 输出 logits 的相对 l2 误差 |
| **Top-1 Agreement ↑** | 与 full-precision 模型 top-1 预测的一致率 |
| **First-prefill Latency** | 生成第一个 token 所需时间 |
| **Speedup ×** | 相对于 BF16 的端到端加速比 |

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **BF16-32** | 全程使用 BF16，作为 full-precision 参考 |
| **BF16-8** | 仅执行 8 步 BF16，测试是否只需少量高精度步即可 |
| **All-FP8** | 全程使用 FP8，代表 uniform quantization 极限 |
| **Early-BF16** | 前 8 步 BF16 + 后 24 步 FP8，与 RAMP 对称但顺序相反（matched-budget control） |
| **Random-20** | 20 种随机混合精度调度（固定 24 FP8 + 8 BF16），用于检验 placement 效应 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（4K context, WikiText & PG19）

#### 表 1：4K 上的 prefill fidelity（部分摘录）
| Schedule | △NLL | Hidden l2 ↓ | Logit l2 ↓ | Top-1 ↑ |
|---------|-------|------------|-----------|--------|
| BF16-32 | 0 | 0% | 0% | 100.00% |
| All-FP8 | +0.00516 | 8.18% | 3.80% | 96.63% |
| Early-BF16 | +0.00444 | 8.15% | 3.78% | 95.95% |
| **RAMP** | **+0.00010** | **2.65%** | **1.34%** | **98.73%** |

> ✅ RAMP 的误差仅为 uniform quantization 的 ~1/40，且远优于 Early-BF16

---

### 端到端性能（NVIDIA H100）

#### 表 2：不同 context length 下的 first-prefill latency 与 speedup
| Context Length | BF16 (ms) | RAMP (ms) | Speedup × |
|---------------|-----------|-----------|----------|
| 128 | 131.784 | 164.972 | 0.7988× |
| 512 | 148.473 | 139.567 | 1.0638× |
| 1024 | 274.090 | 239.395 | 1.1449× |
| 2048 | 543.158 | 469.357 | 1.1572× |
| **4096** | **1103.444** | **949.847** | **1.1617×** |

> ⚡ 在 4K context 下达到 **1.16× 端到端加速**，相当于 **13.9% 的延迟降低**

---

### 消融实验结果

#### （1）精度放置位置的影响（Table 3）
| Schedule | Hidden l2 ↓ |
|--------|------------|
| Random-20 (mean) | 6.17% |
| Early-BF16 | 7.20% |
| **RAMP** | **1.81%** |

> 🔍 RAMP 在所有随机调度中表现最佳，说明“**late high-precision**”具有系统性优势

#### （2）终端 BF16 步数的影响（Table 4）
| FP8 → BF16 Steps | △NLL | Hidden l2 ↓ |
|------------------|-------|------------|
| 32→0 (All-FP8) | +0.00516 | 8.18% |
| 24→8 (RAMP) | +0.00010 | 2.65% |
| 16→16 | -0.00016 | 1.91% |
| 0→32 (Full BF16) | 0 | 0% |

> 📌 结论：仅需最后 **8 步 BF16** 即可恢复大部分精度；继续延长 BF16 收益递减

#### （3）理论预测验证（Figure 1）
- **Figure 1a**：早期注入的扰动（Gaussian / INT8 residual）在最终表示中存活率更低，证实“early error 更易被衰减”
- **Figure 1b**：BF16 步骤的平均深度越高，最终误差越低，RAMP 处于最优端点

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **量化误差具有深度不对称性**：  
   > 早期引入的误差因有更多后续递归步骤进行衰减，对最终表示影响较小；而晚期误差几乎不可恢复。

2. ✅ **最优精度分配策略是“早低晚高”**：  
   > 在固定量化预算下，“前低后高”的 mixed-precision schedule 理论上能最小化最终误差上界（Theorem 3.2）。

3. ✅ **RAMP 显著优于 uniform quantization 和其他 placement 策略**：  
   > 在 4K context 下，RAMP 实现 **1.16× 加速**，同时将 NLL 偏差控制在极低水平（△NLL ≈ 0.0001），远超 All-FP8 和 Early-BF16。

4. ✅ **早期 FP8 步骤并非冗余**：  
   > 对比 BF16-8 控制组可知，仅保留 8 步 BF16 导致性能大幅下降，说明前 24 步 FP8 仍提供了有效的递归 refinement。

5. ✅ **质量随 context 增长保持稳定**：  
   > 在 128–4096 tokens 范围内，RAMP 的 △NLL 和 representation error 无明显增长，表明其适用于 long-context 场景。

---

### 方法的局限性
- ❗ **依赖局部收缩性假设**：理论分析基于 recurrent dynamics 在运行区域内是局部收缩的，若模型动态不稳定可能失效。
- ❗ **仅适用于 prefill 阶段**：未涉及 autoregressive decoding 阶段的优化。
- ❗ **切换点需手动设定**：虽然 24→8 是合理折衷，但缺乏自动化选择 $ K $ 的机制。
- ❗ **对硬件支持有要求**：依赖 H100 等支持 FP8 的设备才能发挥性能优势。

---

### 未来工作方向
- 🔮 探索 **dynamic precision switching**：根据中间表示的敏感度自适应决定何时切换精度。
- 🔮 将 RAMP 思想扩展至 **attention 或其他子模块** 的粒度。
- 🔮 应用于 **其他循环架构**（如 LoopFormer、Universal Transformer）。
- 🔮 结合 **quantization-aware fine-tuning** 进一步压缩误差。
- 🔮 研究在 **decoding 阶段** 是否也可采用类似 temporal mixed-precision 策略。

---

> 💡 **一句话总结**：  
> RAMP 揭示了 recurrent-depth 中“**越早量化越安全**”的本质规律，首次将 mixed precision 按 **time step 分配**，实现了 **高效且高保真** 的 long-context prefill，为 LLM 推理提供了一个新的优化维度。

</details>

---

### 6. [Forgetting is Not Always Bad: A Neuro-Inspired Memory Repair Mechanism for Poisoned LLM Agents](https://arxiv.org/abs/2610.13267)

**Authors**: Lei Liu, Yunji Liang, Xiaowen Zhang, Jingqi Liu, Qi Li, Bin Guo, Zhiwen Yu  
**Category**: cs.AI  
**Published**: 2026-10-13  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.13267v1  

#### Abstract
Large language model (LLM) agents are highly vulnerable to memory-poisoning attacks. Existing defenses primarily rely on external modules for either static memory isolation or continuous online auditing, resulting in low memory efficiency and high computational overhead. To address these limitations...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Forgetting is Not Always Bad: A Neuro-Inspired Memory Repair Mechanism for Poisoned LLM Agents*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（**LLM Agents**）在增强记忆（如通过 **RAG** 构建长期记忆库）后，极易受到**记忆中毒攻击（memory-poisoning attacks）**的威胁。这类攻击通过向记忆库中注入恶意记忆条目，使得这些条目在推理时被优先检索，从而劫持后续决策流程，甚至触发“伪正强化循环”（pseudo-positive reinforcement loop），导致行为持续腐败。

现有防御机制（如 **A-MemGuard**、**LLM Auditor**）依赖外部模块进行在线审计或双存储隔离，存在以下问题：
- **高计算开销**（continuous online auditing）
- **破坏记忆连续性**（dual-memory disrupts reasoning）
- **难以检测休眠的后门触发器**（dormant backdoor triggers）
- **缺乏对异常记忆的动态抑制机制**

### 提出的新方法：DREAM
作者提出 **DREAM**（**Dynamic Reactivation for Engram Attenuation in Memory**），一种受神经科学启发的、**无需训练的、代理内在（agent-intrinsic）**的记忆修复框架，其核心思想是：  
> **通过弱激活（weak reactivation）使恶意记忆进入不稳定状态，并在无强化的情况下促使其衰减——即“功能性遗忘”（selective functional forgetting）**。

#### 创新点
1. **神经科学启发机制**：借鉴生物大脑中“潜意识探测-记忆去稳定化-选择性遗忘”的过程，首次将其应用于 LLM Agent 的安全防御。
2. **三阶段离线修复管道**：
   - **隐式重激活（Implicit Reactivation）**：通过向记忆嵌入添加微弱的各向同性高斯噪声（isotropic Gaussian noise），模拟弱提示，暴露恶意记忆的结构脆弱性。
   - **自适应异常诊断（Adaptive Anomaly Diagnosis）**：基于多维激活模式（multi-dimensional activation patterns）构建诊断分数 `R(t)`，综合衡量几何稳定性（sim）、检索竞争力（pret）和语义一致性（Ecoh）。
   - **动态记忆修复（Dynamic Memory Repair）**：将记忆权重演化建模为离散时间动力系统，可疑记忆指数衰减，良性记忆通过再巩固（reconsolidation）得以保留。
3. **无需训练、轻量高效**：完全不依赖额外训练，仅通过离线扰动和权重更新实现防御，显著降低运行时开销。

### 相比现有方法的优势
| 维度 | DREAM | 现有方法（如 A-MemGuard, LLM Auditor） |
|------|-------|----------------------------------------|
| **效率** | ✅ 低延迟、低 Token 消耗 | ❌ 高计算/Token 开销 |
| **内存连续性** | ✅ 保持单一记忆流 | ❌ 双存储破坏连贯性 |
| **防御机制** | ✅ 内在修复，主动衰减 | ❌ 外部审计，被动过滤 |
| **实用性** | ✅ 无需训练，即插即用 | ❌ 需要额外模型或复杂逻辑 |

---

## 2. 核心实验方法和设置

### 数据集与任务
- **EHRAgent**：医疗记录管理任务，基于 **eICU** 和 **MIMIC-III** 数据集。
- **ReAct-StrategyQA**：知识密集型问答任务，基于 **StrategyQA** 数据集。
- **Multi-Agent System (MAS)**：协作型多智能体系统，使用 **MISINFOTASK** 数据集测试对抗信息注入。

### 攻击类型
- **Backdoor Poisoning**：使用 **AgentPoison [8]**，植入优化触发器。
- **Memory Injection**：使用 **MINJA [9]**，利用语义伪装进行渐进式污染。

### 实验设置
- **LLM Backbone**：`GPT-5.4-mini` 和 `Claude-Haiku-4.5`
- **检索器**：`DPR`（对比学习）和 `REALM`（端到端）
- **防御频率**：每若干轮交互执行一次离线修复周期

### 评估指标
| 指标 | 含义 |
|------|------|
| **ASR-r** | Retrieval-stage Attack Success Rate（检索阶段攻击成功率） |
| **ASR-a** | Reasoning-stage Attack Success Rate（推理阶段） |
| **ASR-t** | End-to-end Task-level Attack Success Rate（端到端任务级） |
| **ACC** | Benign Accuracy（良性任务准确率） |
| **Runtime / Token Consumption** | 运行时间和 Token 消耗（效率指标） |
| **TSR** | Task Success Rate（多智能体任务成功率） |
| **FS** | Final Score（综合错误惩罚得分，越低越好） |

### 基线方法对比
- **No Defense**：无任何防御
- **PPL**：基于困惑度（Perplexity）过滤
- **Distil Classifier**：基于 DistilBERT 的二分类器
- **LLM Auditor**：使用 LLM 审计检索上下文
- **A-MemGuard [17]**：当前最先进的双存储+共识验证防御

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & 2）

#### 在 **AgentPoison** 攻击下的表现（EHRAgent + eICU）
| 方法 | ASR-r ↓ | ASR-a ↓ | ASR-t ↓ | ACC ↑ |
|------|--------|--------|--------|-------|
| No Defense | 100.00 | 100.00 | 100.00 | 70.83 |
| A-MemGuard | 74.87 | 85.31 | 65.24 | 73.85 |
| **DREAM (ours)** | **2.14** | **2.14** | **2.14** | **85.03** |

> ✅ **攻击成功率下降超过 95%**，同时**良性准确率提升至 85.03%**（+14.2%）

#### 在 **MINJA** 攻击下的表现（EHRAgent + eICU）
| 方法 | ASR ↓ | ACC ↑ |
|------|------|-------|
| No Defense | 90.71 | 69.52 |
| A-MemGuard | 14.44 | 58.29 |
| **DREAM (ours)** | **1.60** | **77.01** |

> ✅ **ASR 下降至 1.6%**，**ACC 提升至 77.01%**，远超所有基线

#### 多智能体系统（MAS）表现（Table 3）
| 方法 | FS ↓ | TSR ↑ |
|------|-----|------|
| No Defense | 6.430 | 70.69% |
| A-MemGuard | 6.233 | 89.90% |
| **DREAM (ours)** | **6.038** | **92.96%** |

> ✅ 在协作环境中仍保持最强鲁棒性

#### 效率对比（Fig. 3）
- **速度提升**：相比 A-MemGuard，最高达 **2.13× speedup**
- **Token 消耗减少**：最多减少 **37.7%**
- **绝对开销最低**：仅需 178.42 秒和 46,484 Tokens 处理 10 条查询

---

### 消融实验结果（Ablation Study）

#### 不同激活强度组件的影响（Table 4）
| 组件组合 | ASR-t (EHRAgent) | ACC (EHRAgent) | ASR-a (ReAct) |
|--------|------------------|----------------|---------------|
| 全启用 (`sim`, `pret`, `Ecoh`) | **2.14** | **85.03** | **0.00** |
| 缺失 `pret` | 2.67 | 77.54 | 100.00 |
| 缺失 `sim` | 2.14 | 65.78 | 100.00 |
| 缺失 `Ecoh` | 4.38 | 73.80 | 100.00 |

> 🔍 发现：
> - `pret`（检索竞争力）和 `sim`（几何稳定性）对防止 ReAct 中的推理泄露至关重要。
> - `Ecoh`（语义一致性）对维持 EHRAgent 的高 ACC 至关重要。

#### 不同噪声类型的影响（Table 7）
| 噪声类型 | ACC ↑ |
|---------|-------|
| **Isotropic Gaussian** | **85.03** |
| Uniform | 77.01 |
| Sparse Gaussian | 77.54 |
| Anisotropic Gaussian | 81.28 |

> ✅ **各向同性高斯噪声效果最佳**，因其提供最均衡的局部稳定性探测。

#### 参数敏感性分析（Appendix H）
- **perturbation scale σ**：太小无效，太大扰动过强；最优值在 0.05–0.3 范围内。
- **base decay rate λ₀**：过高会误伤良性记忆，过低则衰减不足。
- **amplification factor k**：适度放大可提升精度，但过大（如 25）会导致 ACC 锐降（32.4%）。
- **recovery rate α_rec**：恢复机制对开放域任务（如 ReAct）尤为关键，关闭后 ACC 从 61.20% 降至 38.00%。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **功能性遗忘是可行且高效的防御机制**：DREAM 成功实现了对恶意记忆的选择性衰减，而无需删除或修改原始参数。
2. ✅ **多维激活诊断优于单信号检测**：结合 `sim`, `pret`, `Ecoh` 显著提升了异常识别能力。
3. ✅ **DREAM 在多种攻击、任务和模型上均表现出色**：无论是 backdoor 还是 injection 攻击，都能将 ASR 压制到接近零。
4. ✅ **显著提升良性性能**：不仅防御有效，还因清理了噪声记忆，反而提高了 ACC（最高 +25%）。
5. ✅ **效率远超现有方案**：在速度和 Token 消耗上全面领先，适合实际部署。

### 方法的局限性（Appendix A）
- **假设攻击者不了解内部探测机制**：若攻击者知晓 `DREAM` 的探测方式（probe-aware attack），可能构造对扰动鲁棒的恶意记忆以规避检测。
- 当前威胁模型下该假设合理（多数攻击为黑盒或 query-only），但未来需考虑更强敌手。

### 未来工作方向
- 扩展到更开放、自由形式的 Agent 场景（如自主探索、长期规划）。
- 探索更复杂的神经动力学模型以进一步优化遗忘策略。
- 研究如何将 DREAM 与其他安全机制（如输入过滤、输出监控）结合，形成纵深防御体系。

---

> 📌 **总结一句话**：  
> DREAM 将“遗忘”从缺陷转化为武器，提出了一种高效、内在、无需训练的记忆修复机制，在几乎不影响良性功能的前提下，实现了对多种记忆中毒攻击的近乎完全防御，为构建可信 LLM Agents 提供了全新范式。

</details>

---

### 7. [SharP: Soft and Hard Prompt-Guided Augmentation with LLM for Low Resource Fake News Detection](https://arxiv.org/abs/2610.13207)

**Authors**: Ziyu Luo, Wei Li, Haoping Huang, Kecheng Liu, Min Gao  
**Category**: cs.CL  
**Published**: 2026-10-13  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.13207v1  

#### Abstract
Fake news detection under low resource conditions is challenged by the scarcity of labeled data and the difficulty of capturing subtle deceptive patterns. In this work, we focus on low resource fake news detection, where only a small labeled set is available for training, while evaluation is perform...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SharP: Soft and Hard Prompt-Guided Augmentation with LLM for Low Resource Fake News Detection

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文聚焦于**低资源环境下的虚假新闻检测**（low resource fake news detection），即在仅有少量标注数据可用的情况下，如何有效提升模型的检测能力。这一场景具有高度现实意义，原因如下：
- 专家事实核查成本高且存在主观歧义；
- 新话题和领域不断涌现，现有基准数据集覆盖不足；
- 传统监督学习难以持续获取大规模高质量标注数据。

现有数据增强方法（如同义词替换、GAN生成、LLM生成）虽然能生成流畅文本，但往往过于通用，缺乏任务相关性，无法保留虚假新闻特有的欺骗性语言模式。

---

### 提出的新方法与创新思路
作者提出 **SharP**（Soft and Hard Prompt-Guided Augmentation with LLM），一种基于大语言模型（LLM）的边界感知文本生成框架，其核心创新包括：

#### （1）**双提示机制（Dual Prompt Mechanism）**
- **Hard Prompt**：提供结构化指令和事实约束（例如“保持原意、最小修改”），确保生成内容语义一致且标签可继承。
- **Soft Prompt**：可学习的连续向量前缀，通过优化注入领域特定的归纳偏置（inductive bias），使LLM更适应虚假新闻的语言风格。
- 二者结合实现对生成过程的细粒度控制，在不微调LLM参数的前提下引导其生成任务相关的合成样本。

#### （2）**边界感知增强策略（Boundary-Aware Strategy）**
- 利用分类器预测熵（predictive entropy）识别决策边界附近的高不确定性样本；
- 主动选择这些“难判样本”进行增强，帮助模型澄清真实与虚假之间的微妙差异。

#### （3）**双重目标优化（Dual Objective Optimization）**
- **目标1（语义保真）**：最大化原始输入与生成文本在特征空间中的相似性（Cosine Similarity）；
- **目标2（判别增强）**：利用分类器反馈损失指导软提示更新，鼓励生成能挑战当前模型的样本；
- 联合优化两个目标，实现生成质量与训练效用的平衡。

---

### 相比现有方法的优势
| 维度 | 优势说明 |
|------|----------|
| **任务对齐性更强** | 不再将LLM视为通用改写工具，而是通过提示工程使其成为任务定制化的增强引擎。 |
| **生成更具信息量** | 针对模型不确定区域生成样本，避免冗余或简单复制，提升训练效率。 |
| **无需额外标注** | 合成样本继承源样本标签，节省人工标注成本。 |
| **参数高效** | 仅优化软提示向量，冻结LLM主干，降低计算开销。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验分为两组：

#### Group A: 事件特异性评估（PHEME）
- 来源：Twitter上的突发新闻事件（breaking-news threads）
- 包含5个子数据集：`Germanwings-crash`, `Sydneysiege`, `Ottawashooting`, `Charliehebdo`, `Ferguson`
- 特点：多模态、对话结构、情感丰富
- 数据划分示例（以Germanwings为例）：
  - 标注假新闻：318条；真实新闻：1481条
  - 未标注池各约140条

#### Group B: 跨基准泛化评估
- **LIAR**：来自PolitiFact的政治声明，细粒度真实性标签 → 二分类（Fake/Real）
- **Twitter15/16**：事件级谣言检测数据集，具回复结构 → 映射为Fake/Real
- 用于验证SharP在不同领域、格式和写作风格下的鲁棒性

> 所有数据集均采用**低资源设定**：从标注集中按比例采样（1%, 5%, ..., 70%）构建训练集，其余固定为测试集。

---

### 实验设置与评估指标
- **评估协议**：固定测试集，变化训练集大小（label budget）
- **评估指标**：Accuracy（Acc）为主，部分分析中也关注Macro-F1
- **主动学习流程**：
  1. 初始化分类器并预训练；
  2. 每轮选取高熵样本（near decision boundary）送入标注；
  3. 使用SharP生成对应增强文本；
  4. 更新训练集并微调分类器；
  5. 迭代多轮直至预算耗尽。

---

### 基线方法对比
| 方法 | 类型 | 简要描述 |
|------|------|---------|
| **HAN** | 层次注意力网络 | 建模词、句层级语义 |
| **dEFEND** | 多模态解释性模型 | 结合文本与用户评论 |
| **HMCAN** | 分层多模态上下文注意力网络 | CNN + 注意力融合多源信息 |
| **DDT** | 基于主动学习的方法 | 动态选择边界样本进行标注 |
| **DeepSeek-7B** | 大规模预训练语言模型 | LoRA微调，代表强基线 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Accuracy）

#### Group A: PHEME 数据集（平均 Acc @ 70%）
| 方法 | Germanwings | Sydney | Ottawa | Charlie | Ferguson | 平均趋势 |
|------|-------------|--------|--------|---------|----------|--------|
| HAN | 0.713 | 0.681 | 0.660 | 0.755 | 0.723 | ~0.70 |
| dEFEND | 0.713 | 0.681 | 0.660 | 0.766 | 0.702 | ~0.70 |
| DDT | 0.927 | 0.906 | 0.906 | 0.885 | 0.927 | ~0.91 |
| **SharP** | **0.938** | **0.906** | **0.927** | **0.927** | **0.948** | **~0.93** ✅ |

> 在所有事件中，SharP均达到最高准确率，尤其在低资源下优势显著。

#### Group B: 跨基准表现（Acc @ 70%）
| 方法 | LIAR | Twitter15/16 |
|------|------|--------------|
| DDT | 0.859 | 0.870 |
| DeepSeek-7B | 0.565 | 0.549 |
| **SharP** | **0.880** | **0.896** ✅ |

> SharP在24种配置中取得21项第一，3项第二，展现出卓越的跨域泛化能力。

---

### 与基线方法的对比结果
- 在极低资源（1%标注）下，SharP相比最佳基线（DDT）平均提升 **+5–10个百分点**；
- 即使在70%标注情况下仍保持领先，表明其不仅适用于小样本，也能持续增益；
- 相比纯LLM微调（如DeepSeek），SharP在低资源下表现远超，体现其数据效率优势。

---

### 消融实验结果（Ablation Study）

![Figure 6: Ablation Study](#)

- **No Augmentation**（仅分类器）：性能随数据增长缓慢，低资源下严重受限；
- **With Augmentation but No Soft Prompt**：有一定提升，但不如完整模型；
- **Full Model (With Soft Prompt)**：在所有比例下均最优，尤其在1%-20%区间提升最明显。

> 表明：**软提示是关键组件**，它使得LLM能够学习到任务特定的生成模式，而非仅依赖硬编码规则。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **LLM可以作为可控的数据增强工具**，但必须通过提示工程（prompting）加以引导，否则易产生泛化无用样本。
2. ✅ **软提示 + 硬提示的协同机制**显著优于单一提示方式，实现了语义一致性与任务相关性的统一。
3. ✅ **边界感知采样策略**极大提升了增强数据的信息价值，让每一轮增强都集中在“最有用”的样本上。
4. ✅ **双重目标优化机制**成功协调了“语义保真”与“判别增强”之间的矛盾，使生成样本既像原文又具挑战性。
5. ✅ SharP在多个异构数据集上表现出色，证明其具备良好的**跨领域泛化能力**。

---

### 方法的局限性
- 当前框架依赖一个预定义的**硬提示模板集合**，可能限制风格多样性；
- 软提示长度需手动调节，过长会影响输入表示；
- 当前实验集中在英文文本，尚未验证在多语言或多模态场景下的有效性；
- 对LLM的选择敏感（文中使用zephyr-7b-beta），不同LLM可能导致性能波动。

---

### 未来工作方向
1. **扩展至其他文本分类任务**：如情感分析、意图识别、主题分类等低资源场景；
2. **支持多语言与跨文化虚假新闻检测**：探索在非英语语境下的适用性；
3. **引入分布外检测（OOD Detection）机制**：防止在未知领域生成误导性样本；
4. **融合多模态信息**：结合图像、用户行为、传播路径等辅助信号进行联合增强；
5. **构建端到端可微分架构**：尝试解决解码阶段梯度截断问题，进一步优化生成路径。

---

> 🔗 **代码开源地址**：[https://github.com/Fujizzz/SharP](https://github.com/Fujizzz/SharP)  
> 📄 **正式版本DOI**：[https://doi.org/10.1016/j.eswa.2026.131178](https://doi.org/10.1016/j.eswa.2026.131178)

</details>

---

### 8. [Collaborative Theory of Mind: Learning to Self-Revise by Anticipating User Responses](https://arxiv.org/abs/2610.12872)

**Authors**: Fanzhi Zeng, Anubrata Das, Maytal Saar-Tsechansky, Liu Leqi  
**Category**: cs.AI  
**Published**: 2026-10-13  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.12872v1  

#### Abstract
Large language models (LLMs) demonstrate strong individual task performance, yet still struggle to collaborate with users over extended, multi-turn interactions, for example by attempting solutions before understanding the user's needs or by failing to revise mistaken assumptions after receiving fee...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Collaborative Theory of Mind: Learning to Self-Revise by Anticipating User Responses**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
当前的 **Large Language Models (LLMs)** 虽然在单轮任务中表现优异，但在多轮协作交互中仍存在显著缺陷：
- 在未充分理解用户需求时就尝试生成解决方案；
- 难以根据反馈修正错误假设；
- 协作效率低，需要用户多次纠正。

这些问题导致即使模型具备强推理能力，也无法实现高效的人机协作。

### **提出了什么新方法或新思路**
本文提出 **Collaborative Theory of Mind (CToM)**，一种通过训练 LLM 主动预测用户响应来改进自身行为的框架。其核心思想是：
- 模型在正式输出前，先**内部起草回复 → 预测用户的可能反馈 → 基于预测进行自我修订**；
- 将“预测用户回应”作为显式的监督学习目标，使模型学会从用户视角审视自己的输出；
- 结合 **Group-Relative Policy Optimization (GRPO)** 进行端到端优化，兼顾任务性能与交互效率。

### **相比现有方法的优势**
| 维度 | CToM 的优势 |
|------|-------------|
| **协作机制** | 不仅被动接受反馈，而是主动预判并提前修复问题，减少用户纠错成本 |
| **训练方式** | 显式建模用户反馈（而非依赖模拟对话或自评），提升预测准确性 |
| **通用性** | 可应用于数学推理、编程、写作等多种任务，无需任务特定设计 |
| **效率提升** | 减少不必要的交互轮次和用户输入量，在相同奖励下更高效 |

---

## 2. **核心实验方法和设置**

### **使用的数据集**
- **Mathematical Reasoning**: 使用 [MATH](https://arxiv.org/abs/2103.03874) 数据集中的 Level-5 竞赛题，涵盖代数、几何、组合等。
- **Code Generation**: 采用 [BigCodeBench](https://arxiv.org/abs/2406.15877)，要求实现满足复杂逻辑的 Python 函数。
- **Document Writing**: 使用 [CollabLLM](https://proceedings.mlr.press/v267/wu25i.html) 的 MediumDocEdit 数据集，基于 Medium 文章进行协同撰写与修改。

> 所有任务均设计为**初始请求不完整（underspecified）**，需通过多轮交互澄清需求。

### **实验设置和评估指标**
#### **模型配置**
- **助理模型（Assistant）**: Qwen3-8B
- **用户模拟器（Simulator）**: Qwen3-32B-FP8，基于真实参考信息提供反馈
- **训练方式**: 结合监督学习（预测损失）与强化学习（GRPO）

#### **评估指标**
| 指标 | 定义 |
|------|------|
| **Task Score** | 数学：正确率（solve rate）；编码：通过单元测试比例；写作：ROUGE-L F1 |
| **Interaction Efficiency** | 用户可见轮数（turns）、用户消息数量及总 token 数（衡量用户努力程度） |
| **Prediction Accuracy** | 预测回复与实际回复之间的词级 F1 和 LLM 判断语义一致性得分 |

#### **停止规则**
- 最大交互轮次限制（math: 4, coding/writing: 6）
- 支持 `COMMIT` 动作提交最终答案

### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **Prompting Base** | 原始 Qwen3-8B + 最小交互提示，无规划、无预测、无修订 |
| **Naive GRPO** | 使用 GRPO 微调 Prompting Base，仅优化任务得分与交互代价 |
| **CToM w/o prediction** | 移除预测监督信号，保留其他结构 |
| **CToM w/o effort** | 移除对用户努力的惩罚项 |
| **CToM task-only** | 同时移除预测监督与努力惩罚 |

---

## 3. **主要实验结果和性能指标**

### **关键性能数据（来自 Table 1）**

| Method | Math Score↑ | Coding Score↑ | Writing Score↑ |
|--------|-------------|---------------|----------------|
| Prompting Base | 69.14 | 34.00 | 13.69 |
| Naive GRPO | 74.08 | 48.91 | 17.92 |
| **CToM (Ours)** | **79.62** | **51.05** | **19.08** |

> ✅ 相比最强基线（Naive GRPO），CToM 在三个领域分别提升：
> - **数学：+7.5%**
> - **编码：+4.4%**
> - **写作：+6.5%**

### **与基线方法的对比结果**
- CToM 在所有任务上均显著优于所有基线；
- 即使 Naive GRPO 允许更多生成预算，仍被 CToM 超越；
- Prompting Base 和 Naive GRPO 往往在第一轮后即终止交互，未能有效利用反馈。

### **消融实验结果（Ablation Studies）**

#### **(1) 预测监督的作用（CToM vs CToM w/o prediction）**
| 指标 | CToM w/o pred | CToM | 差异 |
|------|----------------|-------|-------|
| Math Score | 77.81 | 79.62 | **+1.81** |
| Coding Score | 46.98 | 51.05 | **+4.07** |
| Writing Score | 18.03 | 19.08 | **+1.05** |

✅ 表明**预测监督本身能持续提升任务性能**。

#### **(2) 用户努力成本降低（Table 2）**
| 方法 | Coding Replies↓ | Writing Replies↓ | Coding Tokens↓ | Writing Tokens↓ |
|------|------------------|------------------|----------------|------------------|
| CToM task-only | 3.29 | 4.79 | 82.4 | 104.5 |
| **CToM** | **2.57** | **3.89** | **68.6** | **86.9** |

➡️ **减少用户输入：**
- 编码任务：**回复减少 22%，token 减少 17%**
- 写作任务：**回复减少 19%，token 减少 17%**

> 🔍 分析表明：加入预测监督可在**保持甚至提高任务得分的同时，显著降低用户参与负担**。

#### **(3) 是否提问越多越好？**
- 强制鼓励提问（action-entropy regularization）会导致：
  - 提问频率上升（如写作中 ASK 从 32.7% → 46.1%）
  - 但**任务得分全面下降**（最高降幅达 7.05 pts）
- ❌ 表明“多问”不是万能策略，**智能决策何时该问、何时该改更重要**。

---

## 4. **关键结论和发现**

### **主要发现**
1. ✅ **预测用户反馈可有效提升协作质量**  
   - 模型通过预测用户反应，能在展示前自我修订，避免低效纠错循环。
   
2. ✅ **预测能力可通过显式训练获得**  
   - CToM w/o prediction 的预测准确率远低于完整版，说明预测能力来自专门的监督信号，而非 RL 副产品。

3. ✅ **提升效率而不牺牲性能**  
   - 在相同任务奖励下，引入预测监督可减少用户输入 **8%-17%**，同时提升任务得分。

4. ✅ **形成合理的行为策略**  
   - 模型自动学会在数学/编码任务中较少提问（因信息较明确），而在写作中更频繁提问（因需求高度不确定）。

5. ✅ **内部修订确实有效**  
   - 自我修订增益（self-revision gain）分析显示，CToM 的修订带来更大改进幅度（Figure 3b）。

### **方法的局限性**
- 当前实验依赖**用户模拟器（simulated user）**，尚未验证对真实人类用户的泛化能力；
- 预测目标基于文本匹配，可能忽略深层意图或情感因素；
- 满足条件的“满意”判断由固定规则分类器决定，缺乏灵活性；
- 当前框架主要聚焦于 DRAFT 行动中的修订，ASK 与 COMMIT 决策未充分利用用户模型。

### **未来工作方向**
1. 🧪 **开展真人用户研究**，验证 CToM 在真实场景下的可用性和偏好；
2. 🧠 **扩展 Theory of Mind 的形式**，例如建模用户个性、偏好、情绪状态；
3. 🔄 **将用户模型融入所有决策过程**，不仅用于修订，也指导是否提问、如何提问、何时提交；
4. 📈 **探索更复杂的预测目标**，如多步未来反馈、情感倾向、认知负荷估计；
5. 🔁 **构建闭环迭代系统**，让模型能根据预测误差动态调整其用户心智模型。

---

> 💬 **一句话总结**：  
> **CToM 通过让 LLM 学会“换位思考”——预测用户会怎么回——实现了更聪明、更省力、更高性能的多轮协作，是迈向真正智能协作助手的重要一步。**

</details>

---

### 9. [Attention Sees, MLP Refuses: Tracing the Attention-to-MLP Safety Pathway in LLMs](https://arxiv.org/abs/2610.13078)

**Authors**: Zixuan Weng, Haoqi He, Weifei Jin, Yue Huang, Yang Bai  
**Category**: cs.AI  
**Published**: 2026-10-13  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.13078v1  

#### Abstract
Safety-aligned large language models (LLMs) remain vulnerable to jailbreaks and can over-refuse benign requests. Understanding these failures requires distinguishing harmfulness recognition from the computation that writes refusal. We reframe this as a pathway question and make it measurable on resi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Attention Sees, MLP Refuses: Tracing the Attention-to-MLP Safety Pathway in LLMs*

## 1. 主要贡献和创新点

### 解决的问题
当前对齐的大型语言模型（LLMs）在安全性上存在两大失败模式：
- **Jailbreaks**：恶意提示诱导模型生成有害内容。
- **Over-refusals**：模型对无害请求过度拒绝。

传统行为评测无法区分这些失败是源于模型未能识别危害性（*harmfulness recognition*），还是识别后未能正确执行拒绝（*refusal writing*）。本文旨在从**机制层面**（mechanistic level）解释这一“识别→写作”路径的断裂。

### 提出的新方法与思路
论文提出了一个全新的分析框架，将安全对齐视为一个从 **Attention 到 MLP** 的模块级计算路径，并做出了以下创新：

1. **解耦“识别”与“写作”**  
   首次在 **residual-stream** 层面量化地分离了两个过程：
   - **Harmfulness Recognition**：通过线性探针（linear probe）测量 Attention 和 MLP 更新中可解码的危害性信息。
   - **Refusal Writing**：通过 **Refusal Writing Score (RWS)** 测量每个模块更新对残差流中“拒绝证据”的增量贡献。

2. **提出 Sentinel 诊断与路由系统**  
   基于上述发现，构建了一个双信号诊断系统：
   - **感知分数 P**（Perception Score）：来自 Attention 的危害性识别信号。
   - **决策分数 D**（Decision Score）：来自 MLP 的拒绝写作信号。
   - 利用 P 和 D 的联合分布来诊断 jailbreak 和 over-refusal 的内部机制差异。

3. **实现条件提醒路由（Conditional Reminder Routing）**  
   在生成前，基于 P 和 D 分数动态选择是否插入 `Safety`、`Helpfulness` 或 `Balanced` 提醒，以针对性地纠正不同类型的错误。

### 相比现有方法的优势
- **超越行为评测**：不依赖表面输出，而是深入模型内部机制，提供可解释的故障诊断。
- **优于固定提醒策略**：相比单一的 `Safety` 或 `Helpfulness` 提醒，Sentinel 的动态路由能同时降低 jailbreak 成功率和 over-refusal 率，避免了安全与帮助性的权衡（trade-off）。
- **通用性强**：在 18 对 Base-Instruct 模型（涵盖 Qwen, Llama, Gemma 家族，1B–27B 参数）上验证，具有良好的跨模型泛化能力。

---

## 2. 核心实验方法和设置

### 数据集
- **主要配对数据集**：**JailbreakBench**，提供 100 对语义匹配的有害/无害提示，用于训练探针和分析路径。
- **基准测试集**：
  - **AdvBench**：作为正确拒绝的有害请求参考组。
  - **XSTest** 和 **EvoRefuse**：用于评估 over-refusal，区分正常回答（non-OR）和过度拒绝（OR）样本。
  - **AlpacaEval**：作为正常回答的无害请求参考组。
- **攻击方法**：ReNeLLM, PAIR, AutoDAN, GCG 等主流 jailbreak 攻击生成的对抗提示。

### 实验设置与评估指标

#### 核心度量
| 度量 | 定义 | 作用 |
|------|------|------|
| **Harmfulness Probe AUC** | 线性探针区分有害/无害提示的能力 | 衡量模块更新中的**危害性识别**能力 |
| **Refusal Evidence Score (RES)** | 残差流支持拒绝词（如 "cannot"）vs. 合作词（如 "Sure"）的程度 | 衡量当前状态的**拒绝倾向** |
| **Refusal Writing Score (RWS)** | 模块更新前后 RES 的变化量 | 衡量该模块对拒绝证据的**增量贡献** |
| **Harm-Selective △RWS** | 有害提示与无害提示的 RWS 差值 | 衡量模块是否**选择性地为有害请求写拒绝** |

#### 基线方法对比
- **无提醒（No reminder）**
- **固定提醒**：
  - `Safety reminder`（强化安全）
  - `Helpfulness reminder`（强调帮助性）
  - `Balanced reminder`
- **其他防御方法**：
  - `Self-Reminder` (Xie et al., 2023)
  - `In-Context Defense (ICD)` (Wei et al., 2026)
  - `Llama Guard 3`

#### 评估指标
- **Jailbreak ASR ↓**（Attack Success Rate）：越低越好。
- **Benign False Refusal ↓**：对无害请求的误拒率。
- **Over-refusal False Refusal ↓**：对已知易被过度拒绝的无害请求的误拒率。
- **Aggregate Error (AVG)**：上述三项的算术平均，综合衡量性能。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）路径机制发现
- **早期 Attention 负责识别**：在所有 36 个检查点中，Attention 更新的 **probe AUC 显著高于 MLP**，尤其是在浅层。
- **后期 MLP 负责写作**：在对齐后的 Instruct 模型中，MLP 的 **net refusal writing**（△RWS）远超 Attention。
- **后训练的作用**：Post-training **并未显著提升**危害性识别（AUC），但**大幅增强了 MLP 的拒绝写作能力**，且正向写作更集中在网络深层（最后 1/4 层的贡献从 23.01% 升至 63.86%）。
- **因果干预验证**：重置下游 MLP 更新可消除上游 Attention 干预所引发的拒绝偏移，证实了 **Attention → MLP 的功能链路**。

#### （2）Sentinel 路由性能（Table 1）
在 **8 个模型**上的平均表现（AVG 越低越好）：

| 方法 | Jailbreak ASR ↓ | Benign False ↓ | Over-refusal False ↓ | **AVG** |
|------|------------------|----------------|------------------------|---------|
| No reminder | 95.97 | 0.00 | 88.73 | 61.57 |
| Safety reminder | 30.45 | 2.90 | 85.24 | 39.53 |
| Helpfulness reminder | 68.22 | 0.00 | 59.95 | 42.72 |
| **Sentinel (P+D)** | **35.17** | **0.13** | **55.72** | **30.34** |

✅ **Sentinel (P+D)** 在 **Jailbreak ASR** 和 **Over-refusal** 上均显著优于所有基线，实现了最低的综合错误率。

### 消融实验结果
- **单信号 vs 双信号**：
  - `Sentinel (P only)`：AVG = 38.09%
  - `Sentinel (D only)`：AVG = 34.96%
  - `Sentinel (P+D)`：AVG = **30.34%**
  - ✅ **P 和 D 提供互补信息**，联合使用效果最佳。
- **与固定提醒对比**：
  - `Safety reminder` 虽然 ASR 低（30.45），但 over-refusal 极高（85.24）。
  - `Helpfulness reminder` 降低了 over-refusal（59.95），但 ASR 仍很高（68.22）。
  - ✅ Sentinel 成功打破了这一权衡。

---

## 4. 关键结论和发现

### 主要发现
1. **安全对齐是一个“Attention 看见，MLP 拒绝”的路径**：  
   - **Attention** 模块（尤其是浅层）负责**识别**输入中的危害性。
   - **MLP** 模块（尤其是深层）负责**执行**拒绝动作。
   - 这两个过程是**可分离**的。

2. **后训练（post-training）的本质是增强 MLP 的写作能力**：  
   - 并非从零开始学习识别危害，而是利用已有的识别信号，**强化 MLP 将其转化为拒绝输出的能力**。

3. **Jailbreak 与 Over-refusal 是同一路径两端的故障**：
   - **Jailbreak**：可能是识别失败（P 低）或写作失败（D 低）。
   - **Over-refusal**：表现为识别（P）和写作（D）信号都异常增强。

4. **推理过程中，识别持续而写作延迟**：  
   - 危害性在整个推理过程中都可被解码。
   - 强烈的拒绝写作集中在**答案起始位置**（answer onset），无论是否启用推理。

### 方法的局限性
- **依赖特定的拒绝词集合**：RES 的计算基于预定义的拒绝/合作词表，可能受模型措辞风格影响（尽管作者通过多种构造方式验证了鲁棒性）。
- **仅适用于 Decoder-only 模型**：方法基于 Transformer 的残差流分解，对 Encoder-Decoder 架构不直接适用。
- **动态路由增加计算开销**：虽然复用内部特征，但仍需额外的分类器和路由逻辑。

### 未来工作方向
- **扩展到多模态模型**：探索视觉或音频模态中的“识别→响应”路径。
- **实时干预机制**：基于 P 和 D 信号，在生成过程中进行动态干预，而非仅在开头插入提醒。
- **自动化路径修复**：当检测到路径断裂时，自动调整模型参数或激活模式，实现自适应安全防护。
- **探索其他任务路径**：将此分析框架应用于事实性、诚实性等其他对齐维度。

</details>

---

### 10. [PreSTO: Predictive Subtree Prefetching for Fast LLM Power Sampling](https://arxiv.org/abs/2610.13171)

**Authors**: Nan Jiang, Panagiotis Theodoropoulos, Wei Deng  
**Category**: cs.CL  
**Published**: 2026-10-13  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.13171v1  

#### Abstract
Power sampling improves large language model (LLM) reasoning at test time by sampling from a sharpened version of the base model's distribution. Its sampler, PowerMH, is slow because its Metropolis--Hastings (MH) updates are sequential: each accept/reject decision sets the state from which the next ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PreSTO: Predictive Subtree Prefetching for Fast LLM Power Sampling

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLM）在推理任务中通过 **power sampling** 可以提升性能，该方法通过对基础模型分布进行锐化（sharpening）来增强高概率续写并抑制低概率续写。然而，实现这一目标的主流采样器 **PowerMH** 基于 Metropolis-Hastings（MH）算法，其核心瓶颈在于**序列依赖性**：每个 MH 步骤的接受/拒绝决策必须等待前一步完成才能生成下一个提案（proposal），导致大量串行的模型调用，显著增加了推理延迟。

### 提出的新方法
本文提出了 **Predictive SubTree Prefetching (PreSTO)**，一种用于加速基于 MH 的 power sampling 的框架。其核心思想是将未来的 MH 决策视为一棵由“接受”和“拒绝”分支构成的树，并通过**预取（prefetching）** 来打破序列依赖。

- **Subtree Prefetching**：在每一轮迭代中，PreSTO 构建一个“预取子树”（prefetched subtree），并在一次批量（batched）的模型调用中生成该子树上所有可行的提案。随后，算法沿着实际发生的 MH 接受/拒绝路径执行，从而在一个模型调用中完成多个 MH 转换。
- **Predictive Traversal Rule**：作者观察到，在锐化后的目标分布下，MH 算法只会访问预取子树中一个稀疏且近乎单链（unary）的部分。因此，传统的广度优先搜索（BFS）会浪费预算在不太可能被访问的节点上。为此，作者设计了一种**预测性遍历规则**，该规则利用一个三分类逻辑回归模型，根据候选请求的特征（如 cut fraction）预测其决策属于“确定接受”、“确定拒绝”还是“不确定”的概率，并通过动态规划选择能最大化预期完成 MH 转换数的子树。

### 相比现有方法的优势
- **保持正确性**：PreSTO 在理论上证明了其修改**不改变原始 MH 的转移核（transition kernel）**，保证了输出样本的分布与原始方法等价。
- **效率提升**：通过减少昂贵的串行模型调用次数，换取单次更长的批量调用，实现了显著的速度提升。
- **通用性**：PreSTO 是一个框架，可应用于多种基于 MH 的采样器（如 PowerMH, EntropyCut, MultiTryMH），而不仅仅是修改模型或采样器本身。

## 2. 核心实验方法和设置

### 使用的数据集
实验在七个数据集上进行，覆盖三大类任务：
- **数学推理**：MATH500, AIME 24&25
- **代码生成**：HumanEval, MBPP, LiveCodeBench V6 (LCB V6)
- **科学与通识知识**：GPQA, MMLU

### 实验设置和评估指标
- **基线模型（Base LLMs）**：使用了五个不同的开源 LLM，包括 Qwen3-4B, Qwen3-8B, Qwen3.5-4B, Qwen3.5-9B 和 Gemma4-12B-it。
- **评估指标**：
  - **速度提升（Speedup）**：主要指标，定义为基线方法的平均累计模型调用时间除以 PreSTO 变体的时间。
  - **每轮模型调用完成的 MH 转换数（Transitions per call）**：衡量预取效率。
  - **每次模型调用耗时（Time per call）**：衡量单次批量调用的成本。
  - **样本质量**：通过配对的 TOST 等效性检验（paired TOST equivalence test）比较输出的 token 归一化 log-likelihood 和 confidence 是否与基线等价。
- **超参数**：默认设置为 `K=100` 个 MH 步骤，`Npf=20` 的预取预算，幂指数 `α=4.0`。

### 基线方法对比
PreSTO 框架被应用于以下三种 MH 基线采样器，并与它们的原始版本进行对比：
- **PowerMH**：标准的基于 MH 的 power sampling。
- **EntropyCut**：改进的 PowerMH，根据 token 熵选择切割位置。
- **MultiTryMH**：在每个 MH 步骤中考虑多个候选后缀。

## 3. 主要实验结果和性能指标

### 关键性能数据
在总共 **105** 个（7 数据集 × 5 基础模型 × 3 采样器）的实验设置中，PreSTO 在所有情况下均未变慢，实现了显著的中位数加速：
- 对 **PowerMH** 的中位数加速为 **1.71×**，最大加速达 2.29×。
- 对 **EntropyCut** 的中位数加速为 **1.91×**，最大加速达 2.85×。
- 对 **MultiTryMH** 的中位数加速为 **2.33×**，最大加速超过 7×。

### 与基线方法的对比结果
- **加速来源分析**：加速主要来源于**模型调用次数的大幅减少**。例如，在 LCB V6 上，PreSTO-PowerMH 的调用次数减少了 1.94× 到 4.92×，尽管单次调用时间增长了 1.90× 到 3.42×。
- **样本质量保持**：在 38 次配对的 TOST 等效性检验中，有 **29 次** 通过了检验，表明 PreSTO 输出的 log-likelihood 和 confidence 与基线方法在统计上无显著差异，验证了其正确性。
- **预测性遍历规则的有效性**：在 LCB V6 上使用 Qwen3.5-9B 的实验中，预测性遍历规则相比默认的 BFS 遍历，进一步实现了 **1.25× 到 1.37×** 的额外加速。

### 消融实验结果
- **预取预算（Prefetch Budget）的影响**：随着预取预算 `Npf` 从 8 增加到 20，每轮调用完成的 MH 转换数从 3.59 增加到 4.91，总加速也从 1.20× 提升到 1.44×，证实了更大的预算能带来更好的性能。
- **e-certainty 分布的观察**：实验发现，绝大多数 MH 决策都是“e-certain”（e=0.01），即接受或拒绝的概率极高（>99% 或 <1%）。这解释了为什么预取是有效的——因为决策路径非常确定，预取的大部分工作最终会被丢弃，但通过预测性规则可以更好地分配资源。

## 4. 关键结论和发现

### 主要发现
1.  **预取是加速 MH 采样的有效途径**：通过将未来的 MH 决策建模为一棵树并进行批量预取，可以显著减少串行的模型调用次数，从而降低总延迟。
2.  **MH 路径具有高度确定性**：在 power-sharpened 分布下，MH 算法的路径非常稀疏，大多数决策几乎是确定的。这一洞察是设计高效预取策略的基础。
3.  **智能的资源分配至关重要**：简单的 BFS 遍历会浪费资源在低概率节点上。提出的预测性遍历规则能够更有效地利用预取预算，进一步提升性能。
4.  **正确性与效率可兼得**：PreSTO 在大幅提升效率的同时，通过理论证明和实验验证，确保了其输出分布与原始 MH 采样器一致。

### 方法的局限性
- **内存开销增加**：为了支持批量调用中的前缀缓存（prefix caching），PreSTO 会占用更多的 KV 缓存池。虽然文中指出其峰值占用（约 3%）远低于并行采样方法（如 Best-of-N），但仍是一个需要权衡的因素。
- **依赖于前缀缓存**：PreSTO 的效率高度依赖于现代 LLM 服务引擎（如 vLLM, SGLang）提供的前缀缓存功能来避免重复计算共享前缀。
- **收益存在上限**：由于 MH 路径的确定性，预取的收益并非无限增长，当预取预算过大时，新增加的节点被实际访问的概率极低，导致加速效果趋于平缓。

### 未来工作方向
- **探索更复杂的预测模型**：当前的预测性遍历规则使用简单的逻辑回归，未来可以尝试更强大的模型来更精确地预测 MH 路径。
- **扩展到其他 MCMC 采样器**：PreSTO 框架的思想可以推广到其他具有序列依赖性的 MCMC 采样算法中。
- **优化硬件资源利用率**：进一步研究如何在速度、内存占用和计算成本之间取得最佳平衡，尤其是在大规模部署场景下。

</details>

---

### 11. [Overcoming Data Imbalances with Expert Selection Biases](https://arxiv.org/abs/2610.13370)

**Authors**: Ayush K. Tarun, Clara Meister, Antoine Bosselut  
**Category**: cs.CL  
**Published**: 2026-10-13  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.13370v1  

#### Abstract
Mixture-of-Experts (MoE) models have become the standard architecture for frontier LLMs. The multiple parallel experts in each transformer block theoretically enable expert specialization according to categories such as domains, languages, or modalities during pretraining. However, when data is imba...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Overcoming Data Imbalances with Expert Selection Biases

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **Mixture-of-Experts (MoE)** 架构中，当预训练数据在不同类别（如语言、模态、领域）之间存在严重不平衡时，少数类别的数据往往只能分配到极小部分专家（experts），导致其建模容量受限。这种现象会加剧低资源类别（low-resource categories）的性能退化，尤其是在多语言、多模态或多领域混合训练场景下。

### 提出了什么新方法或新思路
本文提出 **COAX (Category-Oriented Allocation of eXperts)**，一种通过引入**类别条件化的专家选择偏置（expert selection biases）** 来主动引导路由机制的方法。核心思想是：

- 在训练阶段，为每个数据类别（如 Swahili、音频等）指定一个“偏好专家子集”（preferred expert subset）。
- 向该子集中专家的 router 分数添加一个正向偏置 $ \alpha $，从而软性地鼓励同类 token 路由至这些专家。
- 这是一种**软偏置（soft bias）**：所有专家仍可被访问，仅改变路由倾向；且偏置**仅用于训练**，推理时可移除，不增加推理开销。

此外，引入了两个关键控制参数：
- **Preferred expert share ($ q_c $)**：类别 $ c $ 被分配的专家比例。
- **Relative expert share ($ r_c $)**：$ q_c / p_c $，即专家份额与数据份额之比。若 $ r_c > 1 $，表示该类别获得了超出其数据量的专家资源。

### 相比现有方法的优势
| 方法 | 是否显式控制专家份额 | 是否保留跨类别共享 | 推理是否需类别标签 | 性能提升 |
|------|------------------------|--------------------|---------------------|----------|
| **COAX (本文)** | ✅ 显式设定偏好子集大小 | ✅ 软偏置允许覆盖 | ❌ 可移除，无开销 | ✅ 显著提升低资源类别 |
| Hard Routing (如 DEMix) | ✅ 固定专家组 | ❌ 强制隔离 | ✅ 需要 | ⚠️ 无增益甚至下降 |
| Expert Divergence Learning (EDL) | ❌ 间接优化分布差异 | ✅ 保持架构不变 | ❌ 不需要 | ⚠️ 在长尾设置下无效 |

**优势总结**：
- **轻量高效**：仅修改训练时的 router 输入，无需新增模块或改变推理流程。
- **灵活可控**：开发者可直接指定低资源类别的专家配额（如让占 1% 数据的语言使用 4% 的专家）。
- **兼容性强**：可与数据上采样（upsampling）结合，带来额外收益。
- **无推理负担**：偏置可在推理时移除，模型退化为标准 MoE。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验涵盖三大不平衡场景：

#### 多语言（Multilingual）
- **四语言混合**：86.42% 英语 + 10% 韩语 + 2.73% 乌尔都语 + 0.85% 斯瓦希里语
- **八语言混合**：英语 (63.24%) + 中文/韩语各 10% + 克木语/泰米尔语/格鲁吉亚语/乌尔都语 + 斯瓦希里语 (0.85%)

#### 多模态（Multimodal）
- 75% 文本（FineWeb-Edu）  
- 12.5% 视觉（LLaVA-OneVision 图像-文本对）
- 12.5% 音频（Unsupervised People's Speech, GigaSpeech, Libriheavy）

#### 多领域（Domains）
- 85% 通用英文
- 5% 代码（StarCoder）
- 5% 数学（FineMath-3+）
- 2.5% 法律（Legal-mC4）
- 2.5% 医疗（Biomed-Enriched）

### 实验设置和评估指标
- **模型规模**：345M-A147M、3B-A654M、10.6B-A1.2B 三种 MoE 模型（32 层，top-K=8）
- **训练配置**：
  - 所有模型训练 10B tokens（除非特别说明）
  - 使用 Megatron-LM 框架，在 64 块 GH200 GPU 上训练
  - 学习率调度器：WSD（warmup-stable-decay）
- **评估指标**：
  - 主要指标：**验证集 loss 的相对变化（△%）**，负值表示优于基线
  - 辅助指标：ASR 和图像描述任务的 top-k 准确率
  - 功能定位分析：禁用特定专家后观察 loss 变化

### 基线方法对比
- **Unbiased Baseline**：标准 MoE，无任何类别感知路由
- **Hard Routing (DEMix-style)**：强制将某类固定路由到专属专家组
- **Expert Divergence Learning (EDL)**：通过最大化不同类别间的 router 分布 JS 散度来促进专业化
- **Data Upsampling**：传统方法，增加低资源数据采样频率

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 多语言结果（Table 1 & Table 2）
| 模型 | 类别 | COAX △loss (%) | 最佳基线 △loss (%) |
|------|------|---------------|-------------------|
| 345M | Swahili | **-0.55** | -0.59 (with inf bias) |
| 10.6B | Swahili | **-0.30** | -0.41 (with inf bias) |
| 10.6B | 平均（7个少数语言） | **-0.286** | +1.016 (EDL) |

> **趋势**：语言越稀有，提升越大。斯瓦希里语（0.85%）显著受益，而主流语言（英语、中文）基本不受损甚至略有提升。

#### 多模态结果（Table 3）
| 模型 | 任务 | COAX △loss (%) |
|------|------|----------------|
| 345M | I2T（图生文） | **-0.72** |
| 345M | T2I（文生图） | **-0.56** |
| 10.6B | I2T | **-0.25** |
| 10.6B | T2I | **-0.18** |

> 视觉任务提升最一致，音频次之，文本不受负面影响。

#### 多领域结果（Table 4）
- 仅当保留推理偏置时，数学、法律、医疗等领域才显著受益
- **代码** 是唯一在无推理偏置下也持续受益的领域

### 与基线方法的对比结果
| 方法 | 少数语言平均 △loss (%) | 是否优于 COAX |
|------|------------------------|--------------|
| Hard Routing | +0.047 | ❌ 差很多 |
| EDL ($\beta=5\times10^{-4}$) | +1.016 | ❌ 完全失效 |
| **COAX** | **-0.286** | ✅ 最优 |

> **结论**：单纯增强“路由分布差异”（如 EDL）不足以提升性能，必须配合合理的**专家容量再分配**。

### 消融实验结果

#### （1）偏置强度 ablation（Table 6）
- 当推理时不使用偏置时，最优 $\alpha = 0.03$
- 若保留偏置，$\alpha$ 越大越好（最高达 -1.65% 少数语言 loss 下降），但带来更高负载不均衡

#### （2）粗粒度分组有效性（Table 5）
| 分组方式 | 少数语言平均 △loss (%) |
|---------|------------------------|
| 每语言独立设置 | -0.286 |
| **仅两类：英语 vs 其他** | **-0.324** |

> 表明：**分离主导语言与长尾群体** 比精细划分每一类更重要。

#### （3）训练等价性分析（Appendix G.2）
| 模型 | COAX 相当于额外训练多少 tokens |
|------|-------------------------------|
| 345M | **+18.5%** |
| 10.6B | **+2.7%** |

> 即使在最大模型上，COAX 的增益也远超其约 1.3–2.4% 的训练开销，性价比极高。

---

## 4. 关键结论和发现

### 主要发现
1. **类别感知的专家偏置能有效缓解数据不平衡问题**：通过人为扩大低资源类别的“偏好专家池”，可以为其争取更多建模空间。
2. **软偏置优于硬隔离**：完全隔离专家（hard routing）会导致严重的 train-test mismatch，反而损害性能；而 COAX 的软机制既能诱导专业化，又保留必要共享。
3. **多数语言无需精细分区**：简单的“英语 vs 非英语”二分法即可取得良好效果，说明核心在于**减少主干语言对长尾的干扰**。
4. **COAX 与数据上采样互补**：两者结合可进一步放大低资源类别收益（Table 22）。
5. **功能模块化真实发生**：禁用 Swahili 专属专家会使 Swahili loss 上升 107%，而对其他语言影响微弱（Table 11），证明 COAX 成功实现了功能局部化。

### 方法的局限性
- **对某些新引入类别效果有限**：在持续预训练（CPT）中加入俄语、印地语时，COAX 提升不明显（Table 10）。
- **强偏置可能导致推理负担**：若要在推理时保留偏置，则需实时判断输入类别，可能引入延迟。
- **依赖人工定义类别**：当前方法要求预先知道并标注每个 token 的类别，无法处理未知或混合类别。

### 未来工作方向
- 替代手动类别标注 → 探索**自动学习类别结构**（learned structure）
- 结合系统优化 → 设计**拓扑感知的偏置策略**（systems-aware biasing），使偏好专家更匹配 expert-parallel 的硬件布局
- 扩展至动态场景 → 在 streaming 或 online learning 中动态调整偏好专家池

--- 

> 💡 **一句话总结**：  
> COAX 通过在 MoE 路由中引入**可配置的类别偏置**，以极低成本实现了对低资源类别的“专家扩容”，在不改变推理架构的前提下显著提升了长尾性能，是实现公平、高效 MoE 训练的一条实用路径。

</details>

---

### 12. [Mitigating Capability Degradation in LLM Self-Improvement via Representation Geometry Regularization](https://arxiv.org/abs/2610.12855)

**Authors**: Jiaqi Tian, Ting Long  
**Category**: cs.AI  
**Published**: 2026-10-13  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.12855v1  

#### Abstract
On-Policy Self-distillation is a self-improvement paradigm that enables large language models (LLMs) to improve their capabilities by leveraging the same model as both teacher and student under different conditioning contexts. However, we find that OPSD methods can progressively degrade previously a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Mitigating Capability Degradation in LLM Self-Improvement via Representation Geometry Regularization*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
该论文揭示并研究了 **On-Policy Self-Distillation (OPSD)** 方法中存在的一种被忽视的失败模式——**能力退化（capability degradation）**。尽管模型在整体性能上持续提升，但在训练过程中，一些早期能够正确回答的问题会逐渐变为错误回答，即出现“**correct-to-incorrect 转变**”。这种现象威胁到 LLM 自我改进过程的稳定性。

作者通过分析发现，这一退化现象与隐藏层表示（hidden representations）的空间几何结构恶化密切相关：  
- **簇内紧凑性（intra-cluster compactness）下降**  
- **簇间分离度（inter-cluster separation）减弱**

这导致推理路径变得模糊、不一致，最终引发错误输出。

---

### 🛠️ 提出了什么新方法或新思路
为解决上述问题，作者提出了 **Representation Geometry Regularization (RGR)**，一种通用的正则化框架，旨在**在自蒸馏优化过程中保持隐藏表示空间的几何结构稳定**。

RGR 的核心思想是引入两个几何约束：
1. **Intra-cluster Compactness**：鼓励同一语义/推理路径下的隐藏状态保持紧凑；
2. **Inter-cluster Separation**：确保不同推理路径之间的表示具有足够分离度。

这些约束通过控制学生模型与教师模型之间隐藏表示的 **Gram 矩阵差异** 来实现，从而统一地维护表示几何。

此外，RGR 还结合了一个熵正则项（entropy regularization），以进一步稳定输出分布。

最终的整体目标函数为：
$$
\mathcal{L} = \mathcal{L}_{\text{OPSD}} + \alpha \mathcal{L}_{\text{geo}} + \beta \mathcal{L}_{\text{ent}}
$$
其中 $\mathcal{L}_{\text{geo}} = \|HH^\top - H_{\text{ref}}H_{\text{ref}}^\top\|_F$ 是几何保持损失。

---

### ⚖️ 相比现有方法的优势
- **通用性强**：RGR 可无缝集成到多种 OPSD 方法（如 OPSD、SDPO、RLSD、SRPO）中，无需修改其核心优化逻辑。
- **稳定性高**：有效缓解了训练过程中的能力退化问题，减少了“正确→错误”转变的比例。
- **性能增益**：不仅防止退化，还能在多数情况下进一步提升平均准确率（Avg@12）。
- **理论支撑强**：从理论上证明了表示几何对蒸馏风险（distillation risk）的影响，并推导出相应的上界。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **训练数据**：从 OpenThoughts 的数学推理子集中采样最多 30K 个包含思维链（chain-of-thought）的问题-解答对。
- **评估基准**（三大数学竞赛题集）：
  - **AIME 2024**
  - **AIME 2025**
  - **HMMT 2025**  
  （涵盖代数、组合、几何、数论等领域）

---

### ⚙️ 实验设置和评估指标
- **基础模型**：
  - Qwen3-1.7B
  - Qwen3-4B
- **训练配置**：
  - 使用 LoRA（rank=64, α=128）
  - 学习率：5×10⁻⁶
  - GPU：4×NVIDIA A40
  - 所有实验启用 thinking mode
  - 教师模型冻结
- **超参数设置**：
  - α = 2.5×10⁻²
  - β = 5×10⁻²
- **评估指标**：
  - **Avg@12**：每个问题生成 12 条响应，取平均准确率

---

### 🔁 基线方法对比
将 RGR 集成至以下四种代表性 OPSD 方法，并与其原始版本进行比较：
| 方法 | 类型 | 说明 |
|------|------|------|
| **OPSD** | Privileged-info-driven | 使用前向 KL 散度 + token-level KL 截断 |
| **SDPO** | Feedback-driven | 使用 Jensen-Shannon 散度对齐师生分布 |
| **RLSD** | Feedback-driven | 结合环境奖励与 token-level 差异调整更新幅度 |
| **SRPO** | Feedback-driven | 成功轨迹走强化学习，失败走自蒸馏 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）

| 方法 | AIME24 | AIME25 | HMMT25 | Avg |
|------|--------|--------|--------|-----|
| **OPSD** | 56.67 | 43.06 | 28.89 | 42.87 |
| **OPSD + RGR** | **59.89** | **43.60** | **30.00** | **44.50 (+1.63)** |
| **SDPO** | 38.61 | 28.61 | 21.39 | 29.54 |
| **SDPO + RGR** | **41.94** | **31.39** | **21.94** | **31.76 (+2.22)** |
| **RLSD** | 50.27 | 39.70 | 25.83 | 38.60 |
| **RLSD + RGR** | **51.39** | 39.17 | 25.56 | 38.70 (+0.10) |
| **SRPO** | 37.78 | 30.00 | 20.00 | 29.26 |
| **SRPO + RGR** | **39.44** | 29.72 | **21.67** | **30.28 (+1.02)** |

> 在更大规模的 Qwen3-4B 上提升更显著，例如 SDPO + RGR 达到 **59.07 (+3.33)** 的平均增益。

---

### 🔍 与基线方法的对比结果
- RGR 在绝大多数设置下均带来性能提升，尤其在依赖密集 token-level 监督的方法（如 OPSD、SDPO）中效果更明显。
- RLSD 改进较小，因其主要依赖 RL 方向信号，自蒸馏仅用于调制更新幅度，因此表示几何变化较弱。
- RGR 显著降低了“能力退化比率”（见 Figure 3）：
  - **OPSD**：完全消除退化
  - **SDPO**：降低 14.28%
  - **RLSD**：降低 7.15%
  - **SRPO**：降低 21.43%

---

### 🧪 消融实验结果（Ablation Study，Table 2）

| 方法 | AIME24 | AIME25 | HMMT25 |
|------|--------|--------|--------|
| **OPSD** | 56.67 | 43.06 | 28.89 |
| **w/o Ent** | 57.22 | 41.39 | 28.33 |
| **w/o Geo** | 58.61 | 42.22 | 29.17 |
| **RGR (Full)** | **59.89** | **43.60** | **30.00** |

- 移除几何约束（w/o Geo）导致性能显著下降，表明 **representation geometry preservation 是主要贡献因素**。
- 移除熵约束（w/o Ent）也有负面影响，但程度较小，说明其提供的是辅助性稳定作用。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **能力退化普遍存在**：在当前主流 OPSD 方法中普遍存在“正确→错误”转变的现象，即使整体性能上升。
2. **表示几何是关键机制**：隐藏表示的簇内紧凑性和簇间分离性的退化是造成能力丢失的根本原因。
3. **RGR 有效缓解退化**：通过正则化表示几何，RGR 能显著减少能力退化，同时维持甚至增强自我提升能力。
4. **可视化验证**：图示显示，在无 RGR 时，表示空间逐渐混乱；而 RGR 成功保留了清晰的聚类结构（Figure 4）。
5. **跨任务泛化能力**：在 cross-task forgetting 实验中，RGR 明显减轻了灾难性遗忘（如 SDPO 忘记率从 5.28% ↓ 至 1.94%）。

---

### ⚠️ 方法的局限性
- 对某些优化动态较强的算法（如 RLSD），RGR 的增益有限，甚至可能轻微增加遗忘（Table 3 中 RLSD+RGR 忘记率略升）。
- 当前方法依赖于固定教师模型作为参考，未考虑动态演化下的最优几何结构。
- 几何约束基于 Gram 矩阵近似，是一种间接控制方式，可能无法完全捕捉复杂语义结构。

---

### 🔮 未来工作方向
- 探索更自适应的表示几何保持策略，例如动态调整 α 和 β。
- 将 RGR 扩展至其他模型优化范式，如 continual learning、multi-task fine-tuning。
- 研究如何在不依赖教师模型的情况下定义“理想”的表示几何结构。
- 引入可学习的聚类机制，自动识别和保护关键推理路径对应的表示子空间。

---

> ✅ **代码开源地址**：[https://github.com/supermachinelearner/RGR](https://github.com/supermachinelearner/RGR)

</details>

---

### 13. [When Faster Decisions Do Not Make Better Agents: An Empirical Study of Jev in Agent Workflows](https://arxiv.org/abs/2610.13092)

**Authors**: Yuchen Ling, Shengcheng Yu, Zhenyu Chen, Chunrong Fang  
**Category**: cs.AI  
**Published**: 2026-10-13  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.13092v1  

#### Abstract
AI agents depend on repeated decisions to accomplish tasks, with large language models (LLMs) commonly serving as their core decision makers. The recent emergence of Jev offers an opportunity to make these decisions faster and at lower cost. However, it remains unclear whether such local savings can...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*When Faster Decisions Do Not Make Better Agents: An Empirical Study of Jev in Agent Workflows*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文探讨了一个关键问题：**在 AI Agent 工作流中集成更快速、低成本的决策模型（如 Jev）是否能真正提升任务完成的整体效率与可靠性？**  
尽管局部决策成本显著降低，但这些节省能否转化为端到端的任务级收益仍不明确。现有研究多关注单次决策性能，而忽视了决策序列之间的依赖性、恢复机制以及失败尝试带来的资源消耗。

### 提出了什么新方法或新思路
- **系统性实证框架**：首次对 Jev 这类“System One”快速决策模型在多种 Agent 架构中的集成效果进行了大规模、跨场景的实证研究。
- **全流程评估视角**：不仅评估决策层的成本与延迟，还引入了“成功交付结果的单位资源消耗”（resources per delivered outcome）作为核心评价标准，强调从最终任务交付角度衡量价值。
- **行为编码分析**：通过系统性的 trace coding 方法，识别出导致失败的关键行为模式（如 failed recovery, premature completion），揭示性能差异背后的机制。

### 相比现有方法的优势
- **超越局部优化**：指出仅看决策速度或单步成本具有误导性，必须结合辅助推理开销、交互长度、任务成功率等综合判断。
- **揭示反直觉现象**：发现即使每轮尝试花费更低，但由于成功率下降和失败恢复成本高，**平均每次成功交付的成本反而可能上升**。
- **提供设计指导原则**：提出在集成高效决策模型时应重点关注：进度保持（preserving progress）、支持恢复（supporting recovery）、完成验证（verifying completion）三大原则。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **ALFWorld**：基于文本的具身智能环境，模拟家庭任务（如清洁、加热物体）。使用 `valid_unseen` 分割中的全部 134 个任务。
- **AndroidWorld**：移动端自动化基准，模拟在真实 App 中执行操作（如打开 Wi-Fi、添加日历事件）。包含 116 种任务类型，每种固定参数实例运行一次。

### 实验设置和评估指标

#### Agent Configurations
共测试 **5 个 Agent 家族**，每个家族配置多个变体：
| Agent | 类型 | 变体（Variants） |
|-------|------|----------------|
| ReAct | 推理+行动 | B（Baseline）, J（Jev）, C（LLM-Choice） |
| Reflexion | 自我反思 | B, J, C |
| T3A | UI 动作生成 | B, J |
| AutoDroid | 候选事件选择 | B, J |
| FastAgent | 工具调用 | B, J |

其中：
- **B**：原始 LLM 决策策略
- **J**：使用 Jev 进行动作选择
- **C**：控制组，使用 LLM 但输入格式与 Jev 一致（相同候选列表和指令）

#### 执行协议
- 每个配置在相同任务上运行 **3 次**（seed=0）
- 总计 **4,500 次任务执行**
- 使用 `gpt-5.6-luna` 作为主 LLM，`jev-1.13.0` 用于 Jev 决策
- 各 Agent 家族内预算匹配（step/time）

#### 评估指标
| 层面 | 指标 |
|------|------|
| **Effectiveness** | Task success rate, Partial score (AndroidWorld) |
| **Efficiency** | Token usage, Inference cost (mUSD), Elapsed time (s) —— 分为 *per decision* 和 *per task* |
| **Delivery Value** | Cost / time **per delivered success**（含所有失败尝试） |
| **Consistency & Behavior** | 成功率方差、轨迹编码（trace coding）分析失败模式 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### ✅ 决策层级显著增益
- **决策成本下降 66.1%–78.4%**
- **决策延迟减少 73.9%–81.5%**
- 适用于所有五种 Agent，在决策密集型任务中优势明显

#### ⚠️ 任务层级增益有限且不稳定
| Agent | 成本降幅 | 时间降幅 | 成功率变化 |
|-------|----------|----------|------------|
| ReAct | ↓50.9% | ↓47.6% | -1.49 pp |
| Reflexion | ↓32.6% | ↓37.8% | -1.74 pp |
| T3A | ↓17.1% | ↓10.6% | -13.22 pp |
| AutoDroid | ↓68.8% | ↓5.7% | -2.30 pp |
| FastAgent | ↓43.0% | ↑0.7% | -18.10 pp |

> 注：pp = percentage points

#### ❗ 最令人惊讶的结果：**部分 Agent “越省越贵”**
- 尽管每次尝试支出下降，但因成功率降低，**单位成功交付成本上升**：
  - **T3A**：mean cost per success 从 68.70 → **73.97 mUSD**（↑7.7%）
  - **FastAgent**：time per success 从 239.4 → **377.8 秒**（↑58%）

#### 🔍 成功子集对比（matched successes）
当只比较双方都成功的任务时：
- 所有 Agent 均保留显著成本与时效优势
- 表明 Jev 在“走通路径”上确实更快更便宜，但其**失败率更高**抵消了这一优势

### 消融实验结果（J vs C 控制组）
- 在 ALFWorld 上，J-C 对比显示：**大部分性能差距来自接口变更而非模型本身**。
- 使用相同候选列表后（C 组），Jev 的成功率差距缩小至接近零（Reflexion-J vs C：0.00 pp 差距）。
- 说明：**良好的候选构造（candidate construction）是发挥 Jev 潜力的前提**。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **局部效率 ≠ 全局效率**  
   快速决策带来的节省常被以下因素吸收：
   - 辅助推理（auxiliary reasoning）增加
   - 更长的交互序列（additional interactions）
   - 失败后的无效恢复尝试（failed recovery）
   - 过早声明完成（premature completion）

2. **任务成功仍是决定性因素**  
   即使每步更便宜，若无法稳定完成任务，总体资源效益反而恶化。**cost per delivered success 是更合理的评估指标**。

3. **Jev 更适合结构化、目标明确的场景**  
   在 ALFWorld（命令空间固定）中表现稳健；但在 AndroidWorld（UI 动态变化、语义模糊）中失败较多，尤其涉及：
   - 目标识别错误（target identification）
   - 前提条件违反（prerequisite violation）
   - 搜索策略低效（ineffective search）

4. **恢复能力至关重要**  
   Reflexion 实验表明：虽然 Jev 初始成功率略高，但其**恢复初始失败的能力远弱于基线**（11/19 vs 25/26），最终导致总成功率下降。

### 方法的局限性
- **评估范围限制**：仅限语言观察 + 有限动作空间的环境，未覆盖视觉输入、开放工具创建等更复杂场景。
- **系统级耦合影响**：Jev 表现受周围 workflow 设计（如 memory、feedback loop）强烈影响，难以完全归因于模型本身。
- **测量不确定性**：延迟与成本依赖具体服务部署条件，未来模型更新可能改变权衡关系。
- **重复性限制**：三次运行固定 seed，未反映环境或任务分布变化下的稳定性。

### 未来工作方向
- **训练更具后果意识的决策模型**：让 Jev 类模型学习预测选择的长期影响，增强对目标一致性、反馈矛盾的敏感度。
- **动态置信度机制**：引入 calibrated confidence，允许模型在不确定时请求更多信息或拒绝决策。
- **模块化 Agent 架构设计**：将决策建模为可观测组件，显式管理候选输入、输出校验与恢复逻辑。
- **面向部署的评估标准推广**：推动社区采用 *resources per delivered outcome* 而非单纯 success rate 或 step count 作为核心指标。

---

> 💡 **一句话总结**：  
> **Faster decisions don’t automatically make better agents — unless the system knows how to recover from mistakes and verify when the job is truly done.**  
> （更快的决策不会自动造就更好的 Agent——除非系统知道如何从错误中恢复，并确认任务真正完成。）

</details>

---

### 14. [Multi-Stream Attention Residuals](https://arxiv.org/abs/2610.13630)

**Authors**: Xiaohan Qin, Xiangdong Zhang, Tuo Dai, Sunan Zou, Huaijin Wu, Yebin Yang, Zhuo Xia, Lin Yao, Yuliang Liu, Junchi Yan  
**Category**: cs.CL  
**Published**: 2026-10-13  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.13630v1  

#### Abstract
Attention Residuals (AttnRes) enables content-dependent retrieval over network depth, which can be viewed as single-stream softmax attention along the depth axis. Inspired by the success of multi-head attention along the sequence axis, we introduce Multi-Stream Attention Residuals (MSAR), an extensi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Multi-Stream Attention Residuals**

---

## **1. 主要贡献和创新点**

### **解决的问题**
在现代 PreNorm 架构的 Transformer 中，随着网络深度增加，**标准 residual connection** 存在“**depth-wise information dilution**”问题：早期层的信息在深层中被逐步稀释，导致后续层难以有效访问历史表示，限制了模型对深度的高效利用。

尽管已有工作如 **Hyper-Connections (HC)** 和 **Attention Residuals (AttnRes)** 尝试改进，但仍存在以下瓶颈：
- **HC/mHC**：通过多流状态扩展残差路径，但需维护多个并行流，带来存储和通信开销。
- **AttnRes**：引入基于 softmax 的深度注意力机制进行内容依赖检索，但仅限于单头（single-stream）机制，表达能力受限。

### **提出的新方法**
本文提出 **Multi-Stream Attention Residuals (MSAR)**，将 AttnRes 扩展为 **multi-head depth attention** 框架，实现更丰富的深度维度表示检索。

#### **核心设计思想**
- **Contextual Stream Expansion**：  
  利用轻量级的 **multi-scale causal depthwise convolutions** 对单个残差输出生成 $ m $ 个全维流（full-dimensional streams），每个流具有不同的局部上下文感受野（receptive field），从而从序列轴引入多样性。
- **Per-Stream Depth Attention**：  
  每个流独立执行 depth-wise softmax attention，即每个“head”可以基于不同上下文视角选择不同的历史表示。
- **Lightweight Adaptive Mixing**：  
  使用类似 mHC 的 **HPre-style mixer** 将多个 retrieved streams 融合为单一输入，保持计算效率。

### **相比现有方法的优势**
| 特性 | Vanilla Residual | mHC | AttnRes | **MSAR (Ours)** |
|------|------------------|-----|---------|----------------|
| 多流支持 | ❌ | ✅ | ❌ | ✅ |
| 内容依赖检索 | ❌ | ✅ | ✅ | ✅ |
| 隐藏状态完整性保留 | ✅ | ✅ | ✅ | ✅ |
| 单一流存储（低内存） | ✅ | ❌ | ✅ | ✅ |
| 可扩展至多头深度注意力 | ❌ | ❌ | ❌ | ✅ |

> ✅ **关键优势**：MSAR 在不破坏隐藏表示完整性的前提下，实现了真正的 **multi-head attention along the depth axis**，同时通过 on-demand stream expansion 显著降低存储与通信成本。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **预训练语料**：The Pile
- **下游评估基准（共12个）**，分为三类：
  - **General Capabilities**：MMLU, MMLU-Pro, MMLU-Redux, ARC-C, BBH
  - **Code & Math**：HumanEval, LCBench, GSM8K, MATH
  - **Chinese Understanding**：CMMLU, C-Eval, C3

### **实验设置**
- **模型架构**：基于 **DeepSeekMoE-style Transformer**，采用 SwiGLU FFN、Grouped-Query Attention (GQA)、RoPE 位置编码。
- **模型规模**：
  - 主实验：**18B 和 28B MoE models**
  - 消融实验：**10B MoE model**
- **序列长度**：8192
- **训练配置**：
  - Global Batch Size: 768 (28B), 288 (18B)
  - Optimizer: AdamW ($\beta_1=0.9, \beta_2=0.95$)
  - LR Schedule: warmup-stable-decay
- **Block MSAR 设置**：
  - 分块数 $ N \sim 8 $
  - 流数量 $ m = 4 $
  - 卷积核大小：{4, 8, 12}

### **评估指标**
- **下游任务性能**：各 benchmark 上的准确率（%），报告平均分
- **训练效率**：最终 **training loss**
- **计算效率**：**iso-loss compute speedup**（达到相同 loss 所需 compute 的加速比）

### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **Vanilla** | 标准 residual connection |
| **mHC** | Manifold-constrained Hyper-Connections，多流 recurrent state |
| **AttnRes** | Attention Residuals，单流 depth-wise softmax retrieval |
| **MSAR** | 本文方法，multi-stream depth attention |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table 1）**

| Benchmark | 18B Vanilla | 18B MSAR | 28B Vanilla | 28B MSAR |
|-----------|-------------|----------|-------------|----------|
| MMLU | 48.9 | **56.6** | 54.6 | **62.5** |
| MMLU-Pro | 21.1 | **27.9** | 30.1 | **38.7** |
| ARC-C | 55.7 | **74.9** | 70.8 | **78.0** |
| BBH | 32.4 | **38.0** | 41.7 | **46.1** |
| GSM8K | 37.7 | **49.9** | 50.3 | **65.7** |
| HumanEval | 25.6 | **26.8** | 27.4 | **31.1** |
| CMMLU | 42.7 | **50.5** | 47.6 | **56.4** |
| C-Eval | 44.5 | **52.3** | 50.2 | **59.9** |
| **Average** | 37.5 | **45.7** | 44.6 | **52.4** |

> ✅ **结论**：MSAR 在所有 12 个 benchmark 上均显著优于所有 baseline，在 18B 和 28B 规模下分别带来 **+8.2 和 +7.8 的平均提升**。

### **训练损失与计算效率（Figure 1 & Figure 3）**
- **Final Training Loss (18B)**：
  - Vanilla: 1.799
  - mHC: 1.776
  - AttnRes: 1.769
  - **MSAR: 1.750** ✅
- **Scaling Law 结果**：
  - **iso-loss compute speedup**（相对 Vanilla）：
    - mHC: **1.30×**
    - AttnRes: **1.35×**
    - **MSAR: 1.58×** ✅
  > 表明 MSAR 在相同计算预算下可达到更低 loss，或在相同 loss 下节省近 40% compute。

### **消融实验结果（Table 2）**
在 10B MoE 模型上验证关键组件作用（Pile loss ↓）：

| Variant | Pile Loss |
|--------|-----------|
| (1) Vanilla | 1.898 |
| (2) mHC | 1.871 |
| (3) AttnRes | 1.867 |
| (4) AttnRes + mHC (hybrid) | 1.861 |
| **(5) MSAR (default)** | **1.847** |
| (6) w/o Conv (no diversity) | 1.862 |
| (7) w/o HPre (mean fusion) | 1.851 |
| (8) m=2 | 1.850 |
| (9) m=8 | 1.846 |
| (12) S=8 (larger block) | 1.852 |

> 🔍 **发现**：
> - 移除卷积（w/o Conv）导致 loss 上升 0.015 → **source diversity 至关重要**
> - 使用 mean fusion 替代 HPre → loss 上升 0.004 → **adaptive mixing 有额外增益**
> - $ m=4 $ 已接近最优，$ m=8 $ 改进有限 → **收益递减**
> - 更细粒度分块（S=2）略优 → **更多历史源有助于检索**

---

## **4. 关键结论和发现**

### **主要发现**
1. **MSAR 成功实现了 multi-head attention along the depth axis**，突破了 AttnRes 的单头限制。
2. **通过 multi-scale causal convolutions 引入 stream diversity**，无需牺牲隐藏状态完整性。
3. **on-demand expansion 设计使 MSAR 与 AttnRes 共享相同的存储与通信复杂度 $ O(Nd) $**，系统友好。
4. **MSAR 在 18B 和 28B MoE 模型上全面超越 Vanilla、mHC 和 AttnRes**，平均得分提升超 7–8 个百分点。
5. **scaling law 实验表明 MSAR 实现 1.58× 的 iso-loss compute speedup**，显著优于 AttnRes (1.35×) 和 mHC (1.30×)。

### **方法的局限性**
- 当前设计依赖于 **causal convolutions** 来构建上下文视图，可能无法捕捉长距离非局部依赖。
- 虽然 overhead 很小（+0.5% FLOPs, +0.3% 参数），但在极端低延迟场景仍需权衡。
- 多流融合仍为 scalar weighting，未探索更复杂的跨流交互机制。

### **未来工作方向**
- 探索 **non-causal 或 global context-aware stream expansion** 机制。
- 将 MSAR 思想推广到 **vision、multimodal 或 long-sequence modeling** 场景。
- 研究 **dynamic stream allocation**（按需激活不同数量的 stream）以进一步优化效率。
- 结合 **routing 或 sparsity** 机制实现更灵活的 depth-wise information flow 控制。

---

> 📌 **总体评价**：  
> **MSAR 是一种高效且强大的 residual architecture 创新**，它巧妙地结合了 **multi-stream capacity** 与 **content-dependent retrieval** 的优点，为大规模语言模型的深度扩展提供了新的有效路径。其实验充分、设计优雅，具备较强的实用性和推广潜力。

</details>

---

### 15. [Rollout-Time Intervention: Unlocking Agentic Reinforcement Learning Beyond Policy Limits](https://arxiv.org/abs/2610.13438)

**Authors**: Chen Xi, Ding Ruixue, Wang Zhigaoyuan, Zhang Fanrui, Zhang Qiang, Chen Boli, Wang Shihang, Wang Tianfu, Ren Ailin, Wang Pengyong, Xie Pengjun, Xiong Hui  
**Category**: cs.AI  
**Published**: 2026-10-13  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2610.13438v1  

#### Abstract
Reinforcement learning (RL) has emerged as a key paradigm for improving large language model (LLM) agents, enabling stronger reasoning, planning, and tool-use in interactive environments. However, existing agent RL methods remain fundamentally constrained by an on-policy reasoning bottleneck: when t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Rollout-Time Intervention: Unlocking Agentic Reinforcement Learning Beyond Policy Limits

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **on-policy** 的 **Reinforcement Learning (RL)** 在训练 **LLM Agents** 时面临一个根本瓶颈：当任务超出模型当前能力边界时，成功的交互轨迹（successful trajectories）极为稀疏，导致训练信号不足，RL 学习效率低下甚至崩溃。

此外，现有 **teacher-guided** 方法（如 LUFFY、OPD）虽然能引入成功轨迹，但其性能受限于教师模型的能力上限，无法真正突破策略模型自身的推理边界。

### 提出的新方法：Rollout-Time Intervention (RTI)
为解决上述问题，本文提出 **Rollout-Time Intervention (RTI)**，一种在 rollout 过程中动态干预失败轨迹、生成有效学习信号的新范式。RTI 的核心思想是：  
> 即使模型自身无法完成任务，只要能在关键错误步骤进行“校正”，使其后续推理走向成功路径，那么这些“被纠正的成功轨迹”仍可作为有效的学习信号，推动策略超越其原始能力边界。

RTI 包含三个关键技术组件：

1. **Reward-Aware Multi-Agent System (MAS)-Based Intervention**  
   构建由 **Validator-Corrector-Verifier** 组成的多智能体系统，在 rollout 中检测并修正关键错误，同时防止 **privileged information leakage**（如暴露 ground truth 或参考答案）。

2. **Perplexity-Based Distribution Gating**  
   使用 **PPL (Perplexity)** 作为衡量标准，过滤掉与原策略分布偏离过大的干预轨迹，避免因严重 off-policy 偏移导致训练不稳定。

3. **Reasoning-Shift-Aware Token Reweighting**  
   对干预轨迹中的 token 进行动态加权，重点强化那些因干预而发生显著推理转变的关键 token，实现更精准的信用分配（credit assignment）。

### 相比现有方法的优势
- **不依赖更强的教师模型**：RTI 可以使用 **self-intervention**（即用同一模型扮演 Validator/C/Verifer），无需外部更强模型，打破了 teacher-guided 方法的能力天花板。
- **突破策略能力边界**：通过干预生成“本不可能出现”的成功轨迹，使模型能够学习到原本无法探索到的推理模式。
- **稳定高效的学习**：通过 PPL gating 和 token reweighting 机制，确保 off-policy 干预不会破坏训练稳定性。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在五个基准上进行，涵盖真实世界与公开场景：

| 数据集 | 类型 | 描述 |
|-------|------|------|
| **Navigation** | 自建 | 基于高德地图 API 的多途经点导航任务，要求 agent 调用 `poi_search` 和 `direction` 工具规划路线。挑战在于 POI 名称歧义、需上下文消歧、坐标级精确匹配。 |
| **T2-Bench** | 公开 | 多轮对话代理基准，包含 Telecom、Retail、Airline 三个子领域，模拟客服场景下的工具调用与业务逻辑遵循。 |
| **Claw-Eval-Hard20** | 子集构造 | 从 Claw-Eval 中筛选出 20 个极难任务构成的测试集，基础策略几乎无法解决（Pass@3 ≤ 1/3），用于检验探索瓶颈突破能力。 |

### 实验设置与评估指标
- **主干模型**：Qwen3-8B 和 Qwen3.5-9B。
- **训练方式**：基于 **GRPO** 框架进行 RL 微调，group size = 8，binary reward。
- **评估指标**：
  - **Pass@k**：k 次独立 rollout 中至少有一次成功。
  - **Pass\***：k 次 rollout 全部成功。
  - 报告 `p@1`, `avg p@k`, `avg p*` 等聚合指标。

### 基线方法对比
分为三类：

#### (1) Reference Models
- **Vanilla Backbone**：未微调的基础模型。
- **Larger Qwen Variants**：如 Qwen3-235B-A22B、Qwen3.5-397B-A17B。
- **Contemporaneous Proprietary Models**：GPT-4.1 / Claude-4-Sonnet（对应 Qwen3）、GPT-5.1 / Claude-4.6-Sonnet（对应 Qwen3.5）。

#### (2) On-Policy RL
- **GRPO**：标准 group-relative policy optimization。
- **GSPO**：序列级 RL 方法。
- **AEPO**：针对长周期工具使用的熵感知优化方法。

#### (3) Teacher-Guided Methods
- **OPD**：on-policy distillation，教师提供 token-level 评分。
- **LUFFY**：当所有 rollout 失败时，替换为教师生成的成功轨迹。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Qwen3-8B 结果摘要）

| Method | Navigation (p@1) | T2-Telecom (p@1) | Claw-Eval-Hard20 (p@1) |
|--------|------------------|------------------|------------------------|
| Qwen3-8B (Base) | 22.67 | 36.84 | 18.75 |
| GRPO | 35.33 | 36.84 | 22.50 |
| LUFFY | 36.00 | 49.12 | 30.00 |
| OPD | 37.33 | 55.26 | 26.25 |
| **RTI** | **57.33** | **61.40** | **50.00** |

> ✅ **RTI 在所有任务上均显著优于所有基线**，尤其在最难的 Claw-Eval-Hard20 上，Pass@1 达到 **50.00%**，远超第二名 LUFFY 的 30.00%。

### 与基线方法的对比结果
- 在 **Navigation** 上，RTI 比 GRPO 提升 **+22.00 p@1**，说明其在复杂空间推理任务中具有强大优势。
- 在 **T2-Telecom** 上，RTI 比 GRPO 提升 **+24.56 p@1**，展示了其在用户侧工具引导任务中的有效性。
- 在 **Claw-Eval-Hard20** 上，RTI 超越 GRPO **+27.50 p@1**，且远超 teacher-guided 方法，证明其确实突破了 teacher 能力限制。

### 消融实验结果（Ablation Study）
在 Navigation 和 T2-Telecom 上对 RTI 进行消融：

| Variant | Navigation (p@1) | T2-Telecom (p@1) | 说明 |
|--------|------------------|------------------|------|
| RTI (Full) | 57.33 | 61.40 | 完整方法 |
| w/o Post-Check | 26.00 | 38.60 | 性能大幅下降，甚至低于 GRPO，说明 **Verifier 至关重要**，防止信息泄露导致负学习。 |
| w/o PPL Gating | 50.00 | 54.39 | 明显退化，表明 **distribution shift 控制是稳定训练的关键**。 |
| w/o Reweight | 55.33 | 60.53 | 影响较小但仍系统性下降，说明 **token reweighting 提升了 credit assignment 精度**。 |

> 🔍 结论：三大组件缺一不可，其中 **Post-Check（Verifier）** 和 **PPL Gating** 对稳定性影响最大。

---

## 4. 关键结论和发现

### 主要发现
1. **RTI 成功突破了 on-policy RL 的探索瓶颈**：即使模型本身无法完成任务，通过 rollout-time 干预生成“反事实成功轨迹”，也能有效指导策略学习，实现能力跃迁。
2. **self-intervention 足够有效**：使用相同模型进行自我干预即可取得优异效果，无需依赖更强的教师模型，降低了部署门槛。
3. **训练过程是内化的而非依赖干预**：分析显示，随着训练进行，干预轨迹的 **PPL 逐渐降低**，说明模型正在将干预行为“吸收”为自身策略，而非仅在干预时才成功。
4. **促进多步反思性推理**：RTI 训练出的 agent 更倾向于进行多轮查询与验证（如 Navigation 中主动 refine POI 搜索），表现出更强的自省能力。

### 方法的局限性
- **依赖 reward-side evidence**：需要预先定义任务完成的标准（如 evaluation rubrics），不适用于完全开放任务。
- **干预成本增加计算开销**：每步 rollout 需运行 Validator-Corrector-Verifier 流水线，带来额外延迟。
- **对 PPL 阈值敏感**：阈值过高会导致训练不稳定，过低则过滤过多有用轨迹，需仔细调参（文中设为 5 效果最佳）。

### 未来工作方向
- 将 RTI 扩展至 **multi-agent collaboration** 场景，让不同 agent 相互干预。
- 探索 **automated intervention trigger detection**，减少不必要的干预开销。
- 应用于 **embodied agents** 或 **robotic planning**，提升物理世界的任务完成率。
- 研究如何将干预知识 **explicitly distilled** 到 smaller models 中，实现高效迁移。

---

> 📌 **总结一句话**：  
> **RTI 开创了一种“通过干预失败来学习成功”的新范式，使 LLM Agent 能够在不依赖更强教师的情况下，突破自身推理极限，实现真正的 agentic capability expansion。**

</details>

---

### 16. [MENTOR: Learning from Failure with On-Policy Pedagogical Intervention](https://arxiv.org/abs/2610.13594)

**Authors**: Guanlan Yue, Hang Yu, Xiaoyong Huang, Ruipeng Li, Wei Zhang, Bingchang Liu, Rui Wang  
**Category**: cs.AI  
**Published**: 2026-10-13  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2610.13594v1  

#### Abstract
Reinforcement Learning from Verifiable Rewards (RLVR) is a key driver for Large Reasoning Models, but it is vulnerable to "zero-reward collapse", a training failure where a sudden increase in task difficulty causes the learning signal to vanish. We argue that the conventional approach of discarding ...

---

### 17. [RATED: Reliability-Aware AI Agent Tool Retrieval via Execution Dynamics](https://arxiv.org/abs/2610.13889)

**Authors**: Weiliang Chen, Song Liao  
**Category**: cs.AI  
**Published**: 2026-10-13  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2610.13889v1  

#### Abstract
As AI agents increasingly rely on external tools, accurate retrieval is essential for reliable behavior. Existing methods optimize semantic relevance but largely ignore tool availability during execution. To address this gap, we propose RATED (Reliability-Aware Tool Retrieval via Execution Dynamics)...

---

### 18. [Host-Aware Web Crawling for LLM Pretraining Data](https://arxiv.org/abs/2610.13321)

**Authors**: Malte Ostendorff  
**Category**: cs.CL  
**Published**: 2026-10-13  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2610.13321v1  

#### Abstract
Web crawling is a major source of large language model (LLM) pretraining data, yet data-selection pipelines discard a large share of crawled pages. We investigate how a crawler should allocate its budget across and within hosts to collect more high-quality pretraining data. Combining document-level ...

---

### 19. [Proactive Agentic Orchestration](https://arxiv.org/abs/2610.13485)

**Authors**: Xueyang Feng, Chaozhuo Li, Xiaohe Bo, Jiakai Tang, Chenxi Li, Qiwei Ye, Xu Chen  
**Category**: cs.CL  
**Published**: 2026-10-13  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2610.13485v1  

#### Abstract
Agentic orchestration delegates tasks to sub-agents with independent contexts, providing a key path to extending the effective execution horizon of large language model agents. However, existing orchestration methods face two limitations: in execution, current methods fix sub-agents' goals and harne...

---

### 20. [OGPO: Offline Goal-conditioned Policy Optimization for Recoverable Vision-Language-Action Models](https://arxiv.org/abs/2610.12914)

**Authors**: Xule Gao, Xi Wang, Zehua Zang, Rui Wang, Changwen Zheng, Chuxiong Sun  
**Category**: cs.AI  
**Published**: 2026-10-13  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2610.12914v1  

#### Abstract
Vision-language-action (VLA) models have shown strong promise in robotic manipulation, yet their post-training still predominantly relies on trajectory-level supervised imitation over robot demonstrations. While effective for learning feasible actions, this paradigm supervises policies to reproduce ...

---

### 21. [SWEAR: Section-Wise Efficient Audio Reasoning via Self-Rated Difficulty](https://arxiv.org/abs/2610.13081)

**Authors**: Yixuan Li, Minghao Lv, Xuenan Xu, Zihao Zheng, Mengyue Wu  
**Category**: cs.AI  
**Published**: 2026-10-13  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2610.13081v1  

#### Abstract
Recent work on large audio-language models (LALMs) explores adaptive reasoning to balance accuracy and efficiency. However, existing approaches often rely on coarse-grained difficulty estimation, overlooking the diverse factors that determine reasoning effort. We address this limitation by deriving ...

---

### 22. [DRYLAB-Bench: Benchmarking the Biosecurity Risk of Agentic LLM Mutation Design](https://arxiv.org/abs/2610.13116)

**Authors**: Yuzhen Wang, Chunxiao Li, Baichen Le, Lewen Yan, Jie Zhang, Yucheng T. Yang, Xing-Ming Zhao, Lijun Li, Jing Shao  
**Category**: cs.AI  
**Published**: 2026-10-13  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2610.13116v1  

#### Abstract
As large language models (LLMs) acquire advanced biological design capabilities, they offer substantial research benefits but simultaneously introduce severe dual-use risks, such as engineering enhanced pathogens or drug-resistant bacteria. While biosecurity frameworks recognize these threats, the a...

---

### 23. [Quality, Not Just Outcome: A Structural Scoring Framework for Selecting Coding-Agent Trajectories for Supervised Fine-Tuning](https://arxiv.org/abs/2610.13152)

**Authors**: Sida Li, Yuanxin Liu, Shicheng Li, Linli Yao, Xu Sun, Haoli Bai, Xiaohui Li  
**Category**: cs.AI  
**Published**: 2026-10-13  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2610.13152v1  

#### Abstract
Open coding agents are commonly trained through supervised fine-tuning on interaction trajectories whose final patches pass the benchmark test suite. However, this outcome-level filtering criterion ignores process quality: trajectories that eventually succeed may still contain inefficient interactio...

---

### 24. [Training User Simulators with Latent Intrinsic Feedback](https://arxiv.org/abs/2610.13617)

**Authors**: Tarek Naous, Craig Boutilier, Wei Xu, Ofer Meshi  
**Category**: cs.CL  
**Published**: 2026-10-13  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2610.13617v1  

#### Abstract
User simulators are gaining attention as environments to train LLM assistants that can better cope with the nuances of multi-turn conversations with real users. Prior work on user simulation has emphasized generating realistic user utterances, leaving unexplored their potential as a source of direct...

---

### 25. [Enhancing Legal Reasoning in Pre-trained Language Models via Knowledge Graph-Guided Multi-Task Pre-training](https://arxiv.org/abs/2610.13722)

**Authors**: Ibrahim Siddig, Munir Georges  
**Category**: cs.CL  
**Published**: 2026-10-13  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2610.13722v1  

#### Abstract
Legal language understanding requires reasoning over semi-structured regulatory documents, yet standard pre-training objectives do not explicitly target this structural capability. Because current legal language models rely solely on masked language modeling, they treat statutory text as flat token ...

---

### 26. [CLEAT: Learning Value-Guided Clarification for Interactive Agents](https://arxiv.org/abs/2610.12655)

**Authors**: Jiatan Huang, Zheyuan Zhang, Ziming Li, Tianyi Ma, Mingchen Li, Yanfang Ye, Chuxu Zhang  
**Category**: cs.AI  
**Published**: 2026-10-13  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2610.12655v1  

#### Abstract
LLM agents can execute increasingly complex tasks, but effective assistance also requires aligning their actions with goals and constraints that users may leave unstated. We study this problem as interactive intent alignment through clarification. Yet clarification is useful only when the agent can ...

---

### 27. [One-Step Text Generation by Seq-Drifting](https://arxiv.org/abs/2610.12879)

**Authors**: Jiayu Liu, Ruizhi Zhao, Tianyun Ji, Yee Whye Teh, Zhenya Huang, Ning Miao  
**Category**: cs.AI  
**Published**: 2026-10-13  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2610.12879v1  

#### Abstract
Diffusion language models generate tokens in parallel, but still require multiple sampling steps to produce high-quality text. In this paper, we investigate whether a model can learn to generate an entire sequence in a single forward pass. We propose Seq-Drifting, a framework that applies drifting t...

---

### 28. [Collaborative Learning and Reasoning over Model Slices](https://arxiv.org/abs/2610.12927)

**Authors**: Yuheng Wang, Runde Yang, Zhuohan Bao, Qiran Zhang, Huatao Li, Yiyang Zheng, Lin Wu, Jingru Fan, Shu Yao, Yihan Li, Zhi Li, Chen Qian  
**Category**: cs.AI  
**Published**: 2026-10-13  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2610.12927v1  

#### Abstract
Large language models (LLMs) excel across diverse tasks, benefiting from knowledge priors, in-context reasoning, and test-time computation. Despite these strengths, single-model reasoning remains vulnerable to error propagation, as later steps depend on the model's earlier outputs, leaving reasoning...

---

### 29. [AdaC-GRPO: Towards Mastering Long-Horizon Visual Sudoku Puzzles via Adaptive Curriculum Learning](https://arxiv.org/abs/2610.13184)

**Authors**: Sida Li, Yuanxin Liu, Linli Yao, Shicheng Li, Xu Sun, Haoli Bai, Xiaohui Li  
**Category**: cs.AI  
**Published**: 2026-10-13  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2610.13184v1  

#### Abstract
While reinforcement learning with verifiable reward (RLVR) enhances reasoning in large language models, its efficacy in long-horizon multimodal tasks remains underexplored. Visual Sudoku puzzles pose a representative challenge, requiring multi-step deductive reasoning over both local constraints and...

---

### 30. [AlphaNIST: Emergent Assembly Skills from Binary Task Rewards](https://arxiv.org/abs/2610.13354)

**Authors**: Octi Zhang, Iretiayo Akinola, Rosario Scalise, Mateo Guaman Castro, Patrick Yin, Tobias Widmer, Miles Macklin, Chris Amevor, Dylan Turpin, Dirk Van Gelder, Jan Carius, Ankur Handa, Gavriel State, Yunrong Guo, Yashraj Narang  
**Category**: cs.AI  
**Published**: 2026-10-13  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2610.13354v1  

#### Abstract
The NIST Assembly Task Board #1 remains a challenging benchmark for learning-based methods in robotic manipulation. Over short horizons, it demands precise, contact-rich control of small, tight-fitting parts. Over long horizons, it requires recovery and coordination across extended action sequences....

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

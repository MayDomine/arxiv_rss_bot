# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-07-22 08:04:12 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Real-time optimal control with shallow recurrent decoder networks](https://arxiv.org/abs/2607.19302)

**Authors**: Matteo Tomasetto, Francesco Braghin, J. Nathan Kutz, Andrea Manzoni  
**Category**: cs.LG  
**Published**: 2026-07-22  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2607.19302v1  

#### Abstract
Controlling dynamical systems in real-time across multiple scenarios is critical to enabling adaptive control strategies, ensuring stability and efficiency. However, to tailor control actions in response to varying scenarios, traditional optimal control problems typically require several system simu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Real-time optimal control with shallow recurrent decoder networks**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文致力于解决**高维、参数化动力系统在多场景下的实时最优控制**问题。传统方法面临以下挑战：
- **计算成本高昂**：基于 full-order PDE 求解器的最优控制（Optimal Control Problems, OCPs）需要多次仿真，难以满足实时性要求。
- **维度灾难**（Curse of Dimensionality）：状态与控制空间维度极高，尤其在分布式控制中。
- **对模型参数依赖强**：多数 Reduced Order Models（ROMs）需显式输入参数 $\mu$，限制泛化能力。
- **传感器失效风险**：实际应用中传感器可能延迟或故障，影响反馈闭环。

### 🚀 提出的新方法与创新点
提出了一种名为 **SHRED-ROM**（SHallow REcurrent Decoder-based Reduced Order Modeling）的新型数据驱动框架，用于构建**传感器反馈型实时闭环控制器**。

#### 核心创新包括：
1. **基于传感器历史的控制策略学习（Sensor-based Feedback Control）**
   - 利用稀疏、有限的 state sensor readings（如 $N_s \ll N_y$）作为唯一输入。
   - 不依赖显式参数 $\mu$，而是通过时间序列传感器数据隐式编码系统状态与场景信息。
   - 遵循 **Takens 嵌入定理**，利用时滞传感器序列重构高维状态流形。

2. **SHRED 架构作为控制策略（Control Policy）**
   - 使用 **LSTM** 编码器提取传感器时间序列的低维 latent 表示。
   - 使用 **Shallow Decoder Network (SDN)** 将 latent 变量映射到高维控制空间（即预测 $u_k \approx \hat{u}_k$）。
   - 输出可扩展为同时预测控制动作和受控状态（state reconstruction），实现“感知-决策-预测”一体化。

3. **潜空间反馈环（Latent Feedback Loop）应对传感器失效**
   - 引入一个 **latent sensor forecaster**（同样基于 LSTM），以 past sensor values 和 latent 控制变量为输入，预测未来传感器读数。
   - 在传感器缺失时，仍可通过 latent-level 预测维持闭环控制，显著提升鲁棒性。

4. **压缩训练（Compressive Training）提升效率**
   - 对 control/state snapshots 进行 **POD 分解**，仅训练模型预测前 $r$ 个 POD 系数。
   - 显著降低训练复杂度，支持在普通笔记本电脑上完成训练。

### 🔍 相比现有方法的优势
| 方面 | 传统方法 | SHRED-ROM |
|------|--------|-----------|
| 实时性 | ❌ 多次 PDE 求解，耗时 | ✅ 单次前向推理即可输出控制 |
| 参数依赖 | ❌ 需已知 $\mu$ | ✅ 无需显式参数，自动嵌入于传感器历史 |
| 控制形式 | ❌ 多为开环或简化反馈 | ✅ 支持分布式、闭环反馈控制 |
| 数据需求 | ❌ 需大量训练样本 | ✅ 仅需少量专家示范（imitation learning） |
| 容错能力 | ❌ 传感器失效即失控 | ✅ latent forecaster 支持持续运行 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集与案例
论文在三个具有代表性的高维参数化 PDE 控制任务上验证方法有效性：

1. **Fluidic Pinball（流体弹球）**
   - 类型：参数化密度控制（advection-diffusion equation）
   - 参数：三个圆柱旋转速度 $u = [v_1, v_2, v_3]$
   - 目标：控制密度分布避免扩散与边界碰撞
   - 状态维度 $N_y = 7525$，控制维度 $N_u = 59344$

2. **Unsteady Flow Control（非稳态流动控制）**
   - 类型：Navier-Stokes 方程中的边界控制
   - 参数：来流攻角 $\alpha_{\text{in}}$
   - 控制方式：在障碍物前端注入/吸收流体
   - 目标：抑制涡脱落，减少能量耗散与阻力
   - 状态维度 $N_v = 46874$，控制维度 $N_u = 54$

3. **Double Gyre Flow Tracking（双漩涡流跟踪）**
   - 类型：NS 方程中的源项控制（distributed forcing）
   - 参数：扰动振幅 $e$ 与频率 $w$
   - 目标：使流场跟踪动态变化的双漩涡参考流
   - 状态与控制维度均为 $N = 40602$

> 所有数据均通过 **dolfin-adjoint** 求解 full-order OCP 得到最优轨迹，并公开于 Zenodo。

### ⚙️ 实验设置
- **训练样本数**：100–500 条最优轨迹（含 state/control 时间序列）
- **数据划分**：80% 训练 / 10% 验证 / 10% 测试
- **传感器配置**：
  - Fluidic Pinball：1 个移动传感器（随流漂移）
  - Unsteady Flow：3 个固定传感器（测量水平速度）
  - Double Gyre：6 个传感器（3 压力 + 3 水平速度）
- **时滞长度 $L$**：10–25 步，覆盖系统特征时间尺度
- **网络结构**：
  - Encoder（LSTM）：2 层，每层 64 neurons
  - Decoder（SDN）：2 层，350 & 400 neurons，ReLU 激活
  - Dropout rate：0.1
- **优化器**：Adam，初始 lr=1e-3，后半段降为 1e-4，共 200 epochs，batch size=64
- **降维方法**：POD，保留 30–600 个模态用于压缩训练

### 📊 评估指标
- **控制误差**（Mean Relative Error）：
  $$
  e(u,\hat{u}) = \frac{1}{|\mathcal{E}_{\text{test}}|} \sum_{i \in \mathcal{E}_{\text{test}}} \frac{\|u_i - \hat{u}_i\|}{\|u_i\|}
  $$
- **状态重建误差**：
  $$
  e(y,\hat{y}) = \frac{1}{|\mathcal{E}_{\text{test}}|} \sum_{i \in \mathcal{E}_{\text{test}}} \frac{\|y_i - \hat{y}_i\|}{\|y_i\|}
  $$
- **传感器预测误差**：
  $$
  e(s,\hat{s}) = \frac{1}{|\mathcal{E}_{\text{test}}|} \sum_{i \in \mathcal{E}_{\text{test}}} \frac{\|s_i - \hat{s}_i\|}{\|s_i\|}
  $$
- **物理性能指标**：能量耗散、阻力系数、损失函数演化等

### 🔀 基线方法对比
文中未直接列出与其他深度学习控制器的定量比较，但强调其相对于以下类别的优势：
- **传统 ROM + MPC**：需完整物理建模，无法处理强非线性
- **Physics-Informed Neural Networks (PINNs)**：训练慢、易过平滑
- **Deep Operator Networks (e.g., DeepONet)**：需大量控制采样探索空间
- **标准 RNN/Prediction-based MPC**：缺乏 latent-level 容错机制

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

| 任务 | 控制误差 $e(u)$ | 状态重建误差 $e(y)$ | 传感器预测误差 $e(s)$ |
|------|------------------|-----------------------|------------------------|
| Fluidic Pinball | ~5–10%（视觉评估） | — | — |
| Unsteady Flow | **2.68%** | **2.38%** | **0.19%** |
| Double Gyre | **7.47%** | **3.00%** | **0.69%** |

> 注：误差为测试集中平均相对误差，表明模型具备良好泛化能力。

### 🔄 与基线方法的对比结果
虽然没有表格化对比，但从实验分析中可得出：
- **优于开环控制**：SHRED-ROM 能有效响应外部扰动并调整控制策略。
- **媲美 full-order 最优解**：在多个测试场景下，控制效果接近由 adjoint 方法求得的目标解（见 Figure 3–5）。
- **显著优于无控制情况**：在能量耗散、阻力、密度集中等方面均有明显改善（Figure 4, 6, 11）。

### 🔍 消融实验与鲁棒性验证
#### ✅ Latent Feedback Loop 的有效性（Figure 5–7）
- 当从 $t=3$ 秒起关闭所有传感器时，latent sensor forecaster 仍能准确预测后续传感器值（autoregressive mode）。
- 控制性能几乎不受影响，能量耗散与阻力曲线与“始终有传感器”情形高度一致。
- 在高达 **50% 传感器失效率**下，系统仍能稳定运行，性能下降极小（Figure 7）。

#### ✅ 传感器数量与位置不敏感
- 多组实验显示，即使随机布置传感器或更换测量变量（如压力 vs 速度），性能保持稳定。
- 表明方法对 sensor placement 具有较强鲁棒性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **SHRED-ROM 可实现真正的实时闭环控制**
   - 仅需稀疏传感器历史即可在线生成高质量分布式控制指令。
   - 推理速度快，适合部署于边缘设备或安全关键系统（如机器人、航空航天）。

2. **无需显式参数即可泛化至新场景**
   - 参数信息被隐式编码于传感器时间序列中，模型可在训练未见的 $\mu$ 下表现良好。

3. **latent-level sensor forecasting 极大增强鲁棒性**
   - 成功解决了传感器通信中断或故障带来的控制中断问题。
   - 是首个将 latent feedback loop 应用于控制容错的工作之一。

4. **POD 压缩 + SHRED 架构实现高效训练**
   - 即使面对数十万自由度系统，也能在消费级硬件上完成训练。

### ⚠️ 方法的局限性
- **依赖高质量专家示范数据**：当前采用 imitation learning，若专家策略不佳则模型受限。
- **外推能力有限**：在极端参数或未观测动态模式下可能出现偏差。
- **未考虑噪声传感器输入**：目前假设传感器精确，尚未集成 uncertainty quantification。
- **控制可解释性弱**：黑箱神经网络难以提供物理机制解释。

### 🔮 未来工作方向
1. **结合 Reinforcement Learning 进行策略微调**（Fine-tuning）
   - 在预训练基础上进行在线强化学习，适应未知扰动。
2. **引入不确定性量化（Uncertainty Quantification, UQ）**
   - 如使用 ensemble 或 Bayesian 方法判断何时信任控制器。
3. **发展可解释 latent dynamics 模型**
   - 结合 SINDy 或 symbolic regression 发现潜在控制规律。
4. **扩展至更复杂耦合系统**
   - 如 multiphysics、multi-agent 或 hybrid systems。

---

## ✅ 总结
本论文提出了 **SHRED-ROM**——一种基于浅层循环解码器的传感器反馈控制框架，成功实现了**高维、参数化 PDE 系统的实时、鲁棒、闭环最优控制**。其核心在于：
- 利用 **sensor history 替代参数输入**
- 构建 **latent feedback loop 应对传感器失效**
- 实现 **低数据、轻量级、高泛化**的控制策略学习

该方法为智能控制、数字孪生、自主系统等领域提供了强有力的工具，尤其适用于资源受限且安全性要求高的应用场景。

</details>

---

### 2. [Beyond Accuracy and Cost: Latency-Aware LLM Query Routing for Dynamic Workloads](https://arxiv.org/abs/2607.18253)

**Authors**: Shivam Patel, Akaash R. Parthasarathy, Ankur Mallick, Gauri Joshi  
**Category**: cs.AI  
**Published**: 2026-07-22  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.18253v1  

#### Abstract
Modern language query routers improve inference efficiency by assigning each query to a model that balances response quality and monetary cost. However, current query routers are largely latency-agnostic and do not consider the generation latency experienced by queries at model instances. In practic...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Beyond Accuracy and Cost: Latency-Aware LLM Query Routing for Dynamic Workloads*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

现有的 **Language Query Router** 主要关注在 **accuracy** 和 **cost** 之间进行权衡，以高效地将查询路由到合适的 LLM 实例。然而，这些方法普遍是 **latency-agnostic**（对延迟不敏感）的，即它们在决策时忽略了查询在模型实例上实际经历的生成延迟。

与此同时，系统层的负载均衡策略（如 round-robin 或 join-the-shortest-queue）虽然能优化延迟，但完全忽略了模型的准确性和推理成本，导致在 accuracy 和 cost 上表现不佳。

因此，本文旨在解决这一“割裂”问题：  
> **如何在 routing 决策中同时联合优化 accuracy、cost 和 latency？**

---

### 🚀 提出的新方法与创新点

作者提出了一个 **Latency-Aware LLM Query Routing Framework**，其核心创新包括：

#### （1）**Serving Framework Simulation (SFS) 延迟估计器**
- 一种轻量级、高精度的 **Time-to-First-Token (TTFT)** 预测模型。
- 通过模拟现代 LLM serving 框架（如 vLLM）中的 **autoregressive token 批处理过程**，预测新查询在每个候选 model instance 上的等待时间、prefill 时间和首个 decode token 生成时间。
- 显式建模了：
  - 当前 prefill 与 decode 工作负载的混合比例
  - KV-cache 内存管理机制（如 PagedAttention）
  - 调度与批处理策略（如 chunked prefill）
- 支持两种模式：
  - **实时模式**：基于 model instance 的实时 workload 快照进行 SFS 模拟
  - **平均情况模式**：在无实时信息时，使用基于 **Limited Processor Sharing (LPS) 队列模型** 的理论估计

#### （2）**联合优化的路由目标函数**
- 将传统 accuracy-cost utility $ U(\lambda) = \text{acc}_{i,j} - \lambda \cdot \text{cost}_{i,j} $ 扩展为包含延迟项的形式：
  $$
  m(i) \leftarrow \arg\max_{j \in J} \left( U(\lambda) - \delta \cdot L^{\text{ttft}}_{i,j}(t) \right)
  $$
- 或者，在满足 TTFT 约束的前提下最大化 utility：
  $$
  m(i) = \arg\max_{j: L^{\text{ttft}}_{i,j}(t) \leq T_i} U(\lambda)
  $$

#### （3）端到端可集成的轻量设计
- SFS 推理开销极低（平均 **~0.14ms**），远小于典型 TTFT（秒级），适合部署在路由关键路径上。
- 支持动态 workload 变化和 bursty 请求流。

---

### 🔍 相比现有方法的优势

| 维度 | 传统 Query Router | 系统级 Load Balancer | 本文方法 (SFS) |
|------|------------------|------------------------|----------------|
| Accuracy-aware | ✅ | ❌ | ✅ |
| Cost-aware | ✅ | ❌ | ✅ |
| Latency-aware | ❌ | ✅ | ✅ |
| 联合优化 | ❌ | ❌ | ✅ |
| 支持连续批处理（continuous batching）建模 | ❌ | ⚠️（粗粒度） | ✅（细粒度模拟） |

> ✅ **首次实现了 accuracy、cost、latency 三者的联合感知与协同优化。**

---

## 2. 核心实验方法和设置

### 📚 数据集与任务

从四个代表性任务中采样共 **10K 查询（每任务 2.5K）**，覆盖多样化的输入输出长度分布：

| 任务 | 输入长度 | 输出长度 | 特点 |
|------|----------|----------|------|
| **Alpaca** | 短 | 短 | 指令跟随 |
| **GovReport-Summarization** | 长 | 长 | 长文本摘要 |
| **HotpotQA** | 长 | 短 | 多跳问答 |
| **WritingPrompts** | 短 | 长 | 创意写作 |

> 图表显示各任务的 prompt 和 response token 分布广泛（几十至数千 tokens），验证了方法鲁棒性。

---

### 🧪 实验设置

#### 模型池（Model Instances）
使用 Qwen3 系列三个规模模型，部署于不同硬件配置：
- **Qwen3-0.6B** → 单 H100 GPU
- **Qwen3-8B** → 单 H100 GPU
- **Qwen3-32B** → 双 H100 GPU（tensor parallelism）

所有模型均采用 **vLLM** 作为 serving framework，支持 PagedAttention 和 continuous batching。

#### 路由器输入
- **Accuracy Estimator**：LightGBM 回归模型，基于 prompt 特征 + model ID 预测 LLM-as-a-judge 得分（归一化至 [0,1]）
- **Cost Estimator**：基于 token 数 × 定价表（来自 Alibaba Cloud Model Studio）
- **Output Length Predictor**：同样使用 LightGBM 预测 decode token 数量

---

### 📊 评估指标

#### （1）**OnTime Utility**
衡量在满足 TTFT 延迟约束下的平均 accuracy-cost utility：
$$
\text{OnTimeUtility}(\lambda) = \frac{1}{N} \sum_{i=1}^{N} U_{i,m(i)}(\lambda) \cdot \mathbf{1}\{L_{i,m(i)} \leq T_i\}
$$
> 违反延迟约束的请求得分为 0。

#### （2）**Utility-Latency Tradeoff Curve**
通过调节 Lagrangian 系数 $\delta$，绘制 $(\mathbb{E}[U], \mathbb{E}[L^{\text{ttft}}])$ 曲线，比较不同方法的 Pareto 前沿。

#### （3）其他辅助指标
- SLO 达成率（Latency Constraint Attainment）
- 路由分布分析（per-task routing composition）
- 估计误差（MAPE of TTFT prediction）

---

### 🆚 基线方法对比

| 基线方法 | 描述 |
|--------|------|
| **Round Robin** | 循环分配，完全忽略 workload、utility、latency |
| **Shortest Queue** | 路由到 resident queries 最少的实例，仅考虑队列长度 |
| **Latency-Agnostic** | 仅基于最大 $U(\lambda)$ 路由，忽略延迟 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）**OnTime Utility 提升显著**
- 在 Poisson 请求到达场景下（图5）：
  - SFS 方法相比最佳基线（Shortest Queue）实现 **33% 更高的 AUC（Area Under Curve）**
  - 在 5 qps 负载下，**OnTimeUtility 提升达 46%**

#### （2）**Utility-Latency 权衡更优**
- 在固定 $\lambda = 5\times10^{-4}$、varying $\delta$ 的设置下（图6）：
  - SFS 在相同平均 TTFT 下，**utility 提高最多达 40%**
  - 提供平滑可控的 tradeoff 曲线，而基线无法有效平衡三者

#### （3）**延迟估计高度准确**
- SFS 的 TTFT 预测 MAPE（Mean Absolute Percentage Error）仅为 **~5%**
- 对比 throughput-based estimator 的 **85% MAPE**，优势明显（图2）

#### （4）**在 Bursty 请求下依然稳健**
- 使用 **Markov-Modulated Poisson Process (MMPP-2)** 模拟突发流量（burst ratio = 3 和 6）
- SFS 在各种 arrival patterns 下均保持领先（图7），证明其对真实世界动态负载的适应能力

#### （5）**计算开销极低**
- SFS 单次延迟估计耗时 **均值 ~0.14ms**（图8），不影响在线路由性能

---

### 🔬 消融实验与补充结果（见 Appendix G）

- **不同 $\lambda$ 下的一致优势**：无论 accuracy-cost 权重如何变化，SFS 始终优于 Latency-Agnostic 方法（图12）
- **高 SLO 达成率**：SFS 准确预估延迟，避免过载，SLO 满足率始终高于基线（图13）
- **智能路由行为**：
  - 不同任务的路由分布差异显著（图14）
  - SFS 能根据 query 特性选择最优 instance（如长 prompt → 更快小模型；高质量需求 → 大模型）

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Latency-aware routing 至关重要**  
   忽视延迟会导致即使 utility 很高，大量请求仍违反 SLO，最终 OnTime Utility 极低。

2. **SFS 延迟估计器精准且高效**  
   通过对 serving framework 的 token batch 动态进行模拟，能够捕捉 prefill/decode 干扰、KV-cache 状态等复杂因素，显著提升预测准确性。

3. **联合优化带来实质性收益**  
   在相同延迟水平下，SFS 可获得高达 **40% 的 accuracy-cost utility 提升**，或在相同 utility 下显著降低延迟。

4. **方法具有强泛化性**  
   在多种 workload 分布、请求模式（Poisson / MMPP）、$\lambda$ 参数下均稳定优于基线。

---

### ⚠️ 局限性

1. **依赖 workload 快照的可用性**  
   若 model instance 无法提供实时 workload 信息（如第三方云服务），只能退化为平均情况估计，精度下降。

2. **SFS 模拟假设确定性调度策略**  
   对于更复杂的调度器（如优先级抢占、弹性批处理），需相应调整模拟逻辑。

3. **未考虑跨实例通信开销或 placement 优化**  
   当前 focus 在 routing，未联合优化模型部署位置。

4. **实验集中在单一批处理框架（vLLM）**  
   尽管声称可扩展，但在其他框架（如 TensorRT-LLM）上的适配仍需验证。

---

### 🔮 未来工作方向

1. **Joint Optimization of Routing and Placement**  
   同时决定模型部署位置与查询路由，进一步提升资源利用率。

2. **Multi-Renter / Shared Infrastructure Setting**  
   研究多个独立 router 向共享 model instances 发送请求时的博弈与协调机制。

3. **更精细的延迟建模**  
   引入对剩余 decode length 的不确定性建模（如分布预测），提升 bursty 场景下的鲁棒性。

4. **支持 MoE（Mixture-of-Experts）模型的 routing**  
   将方法扩展至 expert-level 路由，结合 sparsity 与延迟控制。

---

## 总结

> **本论文提出了一种全新的 latency-aware LLM query routing 框架，通过引入轻量级的 Serving Framework Simulation (SFS) 延迟估计器，首次实现了 accuracy、cost 与 latency 的联合优化。实验表明，该方法在多种 workload 和请求模式下，相比传统 accuracy-cost 路由器和系统级负载均衡器，可将 OnTime Utility 提升多达 40%，同时保持毫秒级的路由开销，具备良好的实用前景。**

</details>

---

### 3. [Multi-Timescale Latent-Action DRL for Joint Optimization in Edge-Cloud Networks](https://arxiv.org/abs/2607.18288)

**Authors**: Vo Phi Son, Van-Dinh Nguyen, Ngoc Hung Nguyen, Trinh Van Chien, Symeon Chatzinotas  
**Category**: cs.LG  
**Published**: 2026-07-22  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.18288v1  

#### Abstract
Load imbalance across edge and cloud layers degrades latency performance in hierarchical edge-cloud computing (HECC) systems under dynamic task arrivals and heterogeneous resources, leading to severe queuing delays and inefficient resource utilization. To address this challenge, we study a joint ser...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Multi-Timescale Latent-Action DRL for Joint Optimization in Edge-Cloud Networks*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文针对**分层边缘-云计算（HECC）系统中的负载不均衡问题**，该问题在动态任务到达和异构资源条件下会导致严重的排队延迟和资源利用率低下。具体挑战包括：
- **服务放置（Service Placement）**、**用户关联（User Association）**、**计算委托（Computational Delegation）**、**任务卸载（Task Offloading）** 和 **功率控制（Power Control）** 之间的强耦合。
- 决策变量混合离散与连续，导致问题为 **NP-hard 的混合整数非凸优化问题**，传统方法难以求解。

### 提出了什么新方法或新思路
作者提出了一种名为 **2T-MDRL-LA**（Two-Timescale Multi-layer Deep Reinforcement Learning with Latent Action space）的新型深度强化学习框架，其核心创新如下：

- **两阶段时间尺度分解（Two-Timescale Decomposition）**：
  - **长期决策**（Long-term）：处理变化较慢的服务放置、用户关联和计算委托。
  - **短期决策**（Short-term）：适应快速变化的无线信道条件，优化任务卸载和用户发射功率。
  - 通过分离时间尺度，降低联合优化的复杂度并提升稳定性。

- **潜在动作空间设计（Latent Action Space via VAE）**：
  - 引入基于**变分自编码器（VAE）** 的潜在动作表示，将高维组合动作空间压缩到低维连续潜空间。
  - 结合**映射表（Mapping Table）** 将潜空间输出转换为合法的离散系统配置，有效缓解大规模网络中的“维度灾难”。

- **多层 DRL 架构**：
  - 长期子问题采用 **LA-PPO**（Latent-Action PPO），结合 VAE 与 PPO 实现稳定探索。
  - 短期子问题采用标准 **PPO** 处理连续功率控制和二元卸载决策。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **可扩展性** | 潜在动作空间显著降低动作空间维度，支持大规模网络部署。 |
| **收敛速度** | 相比传统 PPO 快约 **50%**，训练更高效。 |
| **性能表现** | 实现接近分支定界（BnB）的**近最优性能**，优于多种 DRL 基线。 |
| **系统适应性** | 能够动态响应负载波动和服务请求变化，维持系统稳定。 |

---

## 2. 核心实验方法和设置

### 实验环境与仿真设置
- **网络拓扑**：在一个 $100 \times 100$ 米区域内部署 $K=2$ 或 $4$ 个 ES（边缘服务器），UEs 随机分布。
- **通信模型**：采用 FDMA 上行链路，带宽 $W=10$ MHz；路径损耗模型为 $PL(d_{mk}) = -35.3 - 37.6\log_{10}(d_{mk})$。
- **计算能力**：ES 计算容量异构（如 [30, 40, 50, 60] GHz），CS 容量为 100 GHz。
- **任务生成**：遵循泊松过程，平均到达率 $\lambda_m \in [10, 18]$ tasks/s。

### 评估指标
| 指标 | 定义 |
|------|------|
| **平均端到端延迟（Average e2e Latency）** | 所有用户任务从生成到完成的平均时间。 |
| **资源利用率（Resource Utilization）** | 实际使用的 CPU 周期占总可用周期的比例。 |
| **任务卸载比例（Task Offloading Ratio）** | 成功卸载至边缘或云端的任务占比。 |
| **全局奖励（Global Reward）** | 负的平均 e2e 延迟，用于衡量策略质量。 |
| **收敛速度** | 达到稳定性能所需的训练步数。 |

### 基线方法对比
| 基线方法 | 描述 |
|--------|------|
| **LA-DDQN-DDQN / LA-DDQN-PPO / LA-PPO-DDQN** | 不同组合的长期/短期算法，验证 PPO 在连续控制上的优势。 |
| **PPO (w/o LA)** / **DDQN (w/o LA)** | 无潜在动作空间的基准，验证 LA 设计的有效性。 |
| **w/o CDO** | 禁用计算委托（Computational Delegation），仅允许本地执行或直接上云。 |
| **w/o SPO** | 固定服务放置，不进行动态优化。 |
| **Random User Association (RUA)** | 用户随机连接 ES。 |
| **Random Processing Task (Rand PT)** | 卸载决策完全随机。 |
| **Branch-and-Bound (BnB)** | 使用 BARON 求解器获得的最优解作为性能上限参考。 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 性能指标 | 数值/提升 |
|---------|----------|
| **平均 e2e 延迟降低** | 最高达 **20.8%**（vs. w/o CDO） |
| **资源利用率提升** | 平均提高 **13%**（vs. w/o CDO） |
| **收敛加速** | 比传统 PPO 快 **约 50%** |
| **与最优解差距（Optimality Gap）** | 相比 BnB 仅差 **约 4%**，实现**近最优性能** |

### 与基线方法的对比结果
- **图 7 & 图 8** 显示，在不同用户数量 $M$ 下，**Alg. 1 (LA-PPO-PPO)** 始终取得最低的平均 e2e 延迟。
  - 当 $M=50$ 时，相比 **w/o CDO** 方案延迟下降 **20.8%**。
  - 相比固定服务放置（w/o SPO）也有明显优势，说明动态服务管理的重要性。
- **图 10** 表明在高任务到达率下（$\lambda_m=18$），所提方法仍保持较低延迟，而 w/o CDO 和 w/o SPO 性能急剧恶化。
- **图 11** 显示随着任务负载增加，**资源利用率持续上升且高于所有基线**，尤其比 w/o CDO 高出 **13%**。

### 消融实验结果
- **图 6(b)** 对比了是否使用潜在动作空间的影响：
  - **LA-PPO** 和 **LA-DDQN** 收敛更快（分别在 ~20k 和 ~25k 步达到饱和）。
  - 未使用 LA 的 **PPO (w/o LA)** 和 **DDQN (w/o LA)** 收敛缓慢（~40k 和 ~35k 步），且最终性能更低。
- **图 5** 显示 **LA-PPO-PPO** 在全局奖励上显著领先，证明了 PPO 在短时连续控制中的优越性。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **计算委托（CDO）对性能至关重要**：启用 ES-ES 和 ES-CS 层面的协同计算可显著缓解局部过载，减少排队延迟。
2. **两阶段时间尺度建模有效解耦复杂决策**：将长期配置与短期调度分离，提升了学习效率和系统稳定性。
3. **潜在动作空间是解决大动作空间的关键**：通过 VAE + Mapping Table 的方式，实现了对高维离散动作空间的高效探索。
4. **2T-MDRL-LA 实现近最优性能**：尽管是启发式方法，但其性能逼近 BnB 解，同时具备实时适应能力。

### 方法的局限性
- **依赖预训练 VAE**：VAE 需要在初始阶段进行训练，可能引入额外开销。
- **映射表大小随网络规模增长**：虽然降低了搜索维度，但映射表本身仍具有 $O(NK^2)$ 复杂度。
- **未考虑安全与隐私因素**：如任务加密、用户数据保护等未纳入优化目标。
- **仿真实验假设理想信息获取**：如全局队列状态、信道增益等，在实际系统中可能受限于观测噪声或反馈延迟。

### 未来工作方向
- 将框架扩展至**移动场景**（UE 移动性建模）。
- 引入**联邦学习机制**以支持分布式训练与隐私保护。
- 探索**在线增量更新机制**，进一步减少重配置开销。
- 考虑**多目标优化**，如联合最小化延迟、能耗与成本。

--- 

> ✅ **总结一句话**：  
> 本文提出的 **2T-MDRL-LA** 框架通过**两阶段时间尺度分解**与**潜在动作空间压缩**，成功解决了 HECC 系统中联合服务放置、计算委托与资源分配的复杂优化难题，在延迟、资源利用和收敛速度方面全面超越现有方法，并逼近最优解。

</details>

---

### 4. [Cross-Dialect Generalization Without Retraining: Benchmarks and Evaluation of Schema-Derived Constrained Decoding for MLIR](https://arxiv.org/abs/2607.18254)

**Authors**: Plawan Kumar Rath  
**Category**: cs.AI  
**Published**: 2026-07-22  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.18254v1  

#### Abstract
Multi-Level Intermediate Representation (MLIR) underlies modern ML compiler infrastructure (TensorFlow, JAX/StableHLO, PyTorch Inductor, IREE), yet appears only in trace amounts in code-LM pretraining corpora. MLIR is also extensible by design: new dialects ship per application domain, so a fine-tun...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Cross-Dialect Generalization Without Retraining: Benchmarks and Evaluation of Schema-Derived Constrained Decoding for MLIR

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代机器学习系统广泛依赖 **MLIR**（Multi-Level Intermediate Representation）作为编译器基础设施（如 TensorFlow、JAX via StableHLO、PyTorch Inductor 等），但 MLIR 在代码语言模型（code-LM）的预训练语料中出现极少。这导致直接使用大模型生成 MLIR 时，常产生语法错误、非法操作符或类型不匹配等问题。

传统方案是为每个 **dialect**（MLIR 的扩展模块）进行微调（fine-tuning），但由于 MLIR 可扩展性强，新 dialect 不断涌现，维护多个微调模型成本高昂且不可扩展。

本文提出：**能否在不重新训练模型的前提下，通过推理时（inference-time）引入结构化先验知识来提升小模型生成合法 MLIR 的能力？**

---

### 提出的新方法与创新思路

作者提出了一个**无需训练、基于 schema 的三层约束解码栈**（schema-derived constraint stack），完全从 MLIR 的 **Operation Definition Specification (ODS)** 自动提取规则，并应用于推理过程：

#### 三层约束栈（C1 + C2 + C3）
| 层级 | 名称 | 功能 |
|------|------|------|
| **C1** | Context-Free Grammar over op signatures | 基于 ODS 自动生成上下文无关文法（CFG），限制只能输出目标 dialect 支持的操作符及其签名格式（如 `arith.addi %a, %b : i32`）。使用 Outlines 库实现 token-level masking。 |
| **C2** | Type-domain grammar splits | 利用 ODS 中的类型约束（如整型 vs 浮点、张量维度一致性等），将 C1 的文法按类型域进一步拆分。例如 `arith.addi` 仅允许整型输入，`linalg.matmul` 要求内维一致。 |
| **C3** | SSA-scope validator | 动态验证 SSA（Static Single Assignment）作用域正确性：确保所有变量使用前已定义且类型匹配。采用五次重试的拒绝采样（rejection sampling）机制。 |

> ✅ **关键创新**：整个约束栈**完全自动化地从 ODS 衍生而来**，无需人工编写规则，因此可无缝迁移到任何提供 ODS 的新 dialect（如从 `arith+func+linalg` 迁移到 `StableHLO` 时未修改一行约束层代码）。

---

### 相比现有方法的优势

| 维度 | 本工作 | 传统方法 |
|------|--------|----------|
| **是否需要训练** | ❌ 无须微调、蒸馏或强化学习 | ✅ 通常需针对特定 dialect 微调 |
| **可扩展性** | ✅ 自动适配新 dialect | ❌ 每个 dialect 需单独建模 |
| **效率** | ✅ 小模型（1.7B）速度快 8–25× | ❌ 大模型（15B–34B）推理慢 |
| **性能表现** | ✅ 在结构主导型 dialect 上超越 34B 模型 | ❌ 大模型仍可能生成非法 IR |

---

## 2. 核心实验方法和设置

### 使用的数据集（四大 NL→MLIR 基准）

首次发布了面向自然语言到 MLIR 转换任务的公开基准，共 **435 个实例**，全部开源（Apache-2.0 协议），附带 [Gebru datasheets](https://dl.acm.org/doi/10.1145/3458754.3469731) 和 [Croissant 1.0](https://mlcommons.org/en/croissant-v1/) 元数据。

| 数据集 | 规模 | Dialect | 类型 | 特点 |
|-------|-----|---------|------|------|
| **MLIR-Spec-150** | 150 | arith+func+memref | 手工构建 | 混合难度，覆盖基础运算 |
| **Linalg-Spec-30** | 30 | linalg | 手工构建 | 聚焦 12 个核心线性代数操作 |
| **StableHLO-Spec-30** | 30 | StableHLO | 手工构建 | 覆盖 10 个算子族 |
| **StableHLO-Held-Out-200** | 200 | StableHLO | 参数化生成 | 模板化 sweep 构造，避免作者偏见 |
| **StableHLO-Out-Of-Grammar-25** | 25 | StableHLO | 手工构建 | 测试超出文法范围的行为 |
| **Functional Reference Set (n=30)** | 30 | 多 dialect | 手工构建 | 提供输入/输出对，用于功能正确性测试 |

---

### 实验设置

- **主模型**：`SmolLM2-1.7B-Instruct`（fp16，运行于 Apple M4 Max）
- **基线模型**：
  - `CodeLlama-34B`
  - `Granite-Code-34B`
  - `StarCoder2-15B`
  （均通过 Ollama + llama.cpp 加载，Q4_K_M 量化）

- **评估协议**：
  - 所有模型统一采用 **五次重试拒绝采样**（five-retry rejection sampling）
  - Prompt 格式固定（三样本提示，3-shot priming）
  - 评价指标以 **verify-valid rate** 为主（即能通过 `mlir-opt` 或 `iree-compile` 验证）

- **统计方法**：
  - 报告三随机种子下的平均值 ± 半区间（half-range）
  - 使用 **paired bootstrap**（10,000 次重采样）计算 95% 置信区间（CI）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 3）

| System | arith+func (n=200) | linalg (n=125) |
|--------|-------------------|---------------|
| **SmolLM2-1.7B + C1+C2+C3** | 53.2% (±1.8pp) | **80.0%** (±0.0pp) |
| CodeLlama-34B + C1+C3 | 59.8% (±0.8pp) | 58.7% (±2.4pp) |
| Granite-Code-34B + C1+C3 | 51.5% (±2.5pp) | 35.7% (±4.4pp) |
| StarCoder2-15B + C1+C3 | 66.8% (±1.5pp) | 54.9% (±1.2pp) |

> 🔥 **核心亮点**：  
> 在 **linalg** 上，**1.7B 的 SmolLM2 以 80.0% 的 verify-valid 率全面击败所有 15B–34B 基线模型**（领先 21–44 个百分点，置信区间无重叠）！

---

### 与其他基线的对比结果

| 场景 | 结果 |
|------|------|
| **linalg** | ✅ **显著胜出**：SmolLM2 超越所有 34B 模型，且三次种子均为 80.0%，结果稳健 |
| **StableHLO-Spec-30**（手工构造） | ✅ 胜出：63.3% vs 33.3–36.7%（34B 模型） |
| **StableHLO-Held-Out-200**（参数化模板） | ❌ 失败：SmolLM2 61.5% vs 基线 98–100%（说明大模型在模板任务上已饱和） |
| **arith+func** | ⚠️ 平局至略输：53.2% vs 最高 66.8%（StarCoder2-15B） |

> 📌 **结论具有 dialect 条件性**：该方法在 verifier 语义由**结构约束主导**的 dialect（如 linalg）上效果最好；而在 attribute-value 决定合法性（如 arith+func）的任务上优势减弱。

---

### 消融实验结果（Ablation Study）

| 消融项 | 效果（linalg） | 效果（arith+func） |
|--------|----------------|--------------------|
| **C1 → C1+C2** | +4.0pp (p=0.006) | 0.0pp |
| **C1+C2 → C1+C2+C3** | +0.8pp（接近零） | **+13.0pp** (p < 0.0001) |

> 💡 **解读**：
> - **C2 对 linalg 至关重要**：因为其类型结构复杂（如矩阵乘法维度匹配），类型域拆分有效过滤非法组合。
> - **C3 对 arith+func 更关键**：因 SSA 作用域错误更常见，动态验证大幅提升成功率。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **小模型 + schema 约束 ≈ 甚至 > 大模型自由生成**
   - 在结构敏感型 dialect（如 `linalg`）上，`SmolLM2-1.7B` 在推理时加入 C1+C2+C3 后，**性能超越 15B–34B 开源 code-LM**。
   - 推理速度达 **8–25× 加速**（单次生成 1.65–1.86 秒 vs 16–40 秒）。

2. ✅ **约束栈可跨 dialect 机械迁移**
   - 从 `arith+func+linalg` 迁移到 `StableHLO` **无需修改任何约束层代码**，证明了 schema-driven 方法的通用性和可扩展性。

3. ✅ **verify-valid 是必要但非充分条件**
   - 功能正确性（output-match）远低于结构正确率。例如在 linalg 上：
     - verify-valid: 80.0%
     - output-match: 仅 20.0%
   - 差距主要源于静态形状包装器（wrapper）不一致，而非逻辑错误。

4. ✅ **“Hidden Cost of Structure”现象被缓解**
   - 以往研究发现约束解码有时会降低小模型表现（Schall & de Melo, 2025），但在本文设置下（三样本提示 + 类型感知文法）未观察到此现象，表明现代工程设计可吸收该问题。

---

### 方法的局限性

| 限制 | 描述 |
|------|------|
| **arith+func 性能平庸** | 在 attribute-value 主导的 dialect 上无法超越大模型，视为“非胜利单元”（non-win cell） |
| **StableHLO 结果依赖数据集** | 在手工构造集（Spec-30）上领先，但在参数化集（Held-Out-200）上落后，说明大模型在模板任务上已达饱和 |
| **基线模型约束较弱** | 34B 模型的 C1 是基于解析的拒绝采样，弱于 SmolLM2 的 token-level masking，存在不对称性 |
| **仅测试单一 SLM 家族** | 主要使用 SmolLM2，未探索其他小模型家族 |
| **inline coupled decoder 未实用化** | 理论上等价于拒绝采样，但实际实现易触发 `max_tokens` 限制，尚未可用 |

---

### 未来工作方向

1. **闭合结构与功能之间的鸿沟**
   - 构建更大规模的功能正确性基准（functional correctness benchmark）
   - 引入自动 lowering + 执行 + 输出比对流程

2. **开发 C4：形状推断感知约束层**
   - 当前残余错误多来自 `transpose`, `broadcast` 等 shape-reasoning 操作
   - 可利用 ODS 中的 `InferShapedTypeOpInterface` 提取形状函数，构建动态形状约束

3. **完善 inline coupled decoder**
   - 当前版本因 BPE 分词边界问题难以收敛
   - 需支持回溯机制以提高采样效率

4. **扩展文法覆盖范围**
   - 当前支持 `arith`, `func`, `memref`, `linalg`, `StableHLO`
   - 可轻松扩展至 `scf`, `affine`, `tensor`, `vector` 等 dialect

---

## 总结

> 🏆 **一句话总结**：  
> 本文展示了**无需训练的小模型 + schema-derived 约束解码**，可在结构敏感型 MLIR dialect（如 linalg）上**以极高速度超越 34B 级大模型**，并发布首个公开的 NL→MLIR 基准套件，推动 MLIR 生成领域的标准化评估。

> 🔧 **核心价值**：  
> 提供了一种**低成本、高可扩展、免训练**的方式，让小模型也能胜任复杂中间表示（IR）生成任务，特别适合快速迭代的新领域编译器开发。

</details>

---

### 5. [What Governs Decode Throughput in Absolute-Offset GPU LZ77? A Work-Granularity Mechanism and an Encode-Time Min-Match-Length Lever](https://arxiv.org/abs/2607.18541)

**Authors**: Yakiv Shavidze  
**Category**: cs.DC  
**Published**: 2026-07-22  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.18541v1  

#### Abstract
The ACEAPEX line of work established a lossless LZ77 format whose back-references are absolute output positions, giving parallel, compressed-resident GPU decode with sub-millisecond region seek. What it did not establish is what governs the decode throughput of such a format, or how to improve it. T...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*What Governs Decode Throughput in Absolute-Offset GPU LZ77? A Work-Granularity Mechanism and an Encode-Time Min-Match-Length Lever*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

本文解决了 **absolute-offset GPU LZ77 解码吞吐量受限的根本原因** 这一未被充分研究的问题。尽管已有工作（如 ACEAPEX 系列）实现了在 GPU 上对压缩数据的并行、随机访问解码，但这些工作并未解释“**是什么决定了 decode throughput**”，也未提供有效的优化杠杆。

具体而言，作者探究了以下疑问：
- 是 occupancy、compute、address scatter，还是 launch parallelism 限制了解码速度？
- 如何在不修改 decode kernel 的前提下提升性能？

### 🚀 提出了什么新方法或新思路

#### （1）提出 **work granularity 机制** 作为决定性因素
- 发现解码吞吐量主要由 **平均匹配长度（average match length）** 决定，即 **work granularity**。
- 短匹配导致 GPU warp 中多数线程空闲（underloaded），造成资源浪费；长匹配则能更好利用并行性。

#### （2）提出 **encode-time min-match-length lever**
- 在编码阶段提高不同距离类别的最小匹配长度（min-match-length）阈值：
  - 原始：`6/8/10/12`
  - 调优后：`12/16/24/32`（按距离分段）
- **无需改动 decode kernel**，即可同时提升 **解码吞吐量** 和 **压缩率（compression ratio）**

> 💡 创新之处在于：这不是传统意义上的“权衡”（trade-off），而是通过移除低效短匹配实现**双赢**——既减少熵开销，又改善 GPU 并行利用率。

### 🔍 相比现有方法的优势

| 方面 | 优势说明 |
|------|----------|
| **机制理解深度** | 首次系统性地通过 controlled ablation 实验排除多种假设（occupancy、scatter 等），确立 work granularity 的主导作用 |
| **优化方式轻量高效** | 所有增益均来自 encode-side 调整，decode kernel 完全不变，部署成本极低 |
| **双重收益** | 同时提升 decode throughput 与 compression ratio，打破“压缩越强、解码越慢”的直觉 |
| **bit-perfect 可验证** | 所有结果均可复现，GPU 路径使用 FNV 校验，CPU 路径字节比较，确保正确性 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集

| 数据集 | 描述 |
|--------|------|
| **FASTQ 1GB** | NA12878 测序数据（真实基因组数据） |
| **enwik9** | Wikipedia XML 转储的前 1GB，常用于压缩基准测试 |
| **Silesia corpus** | 12 个小型通用文件集合，广泛用于压缩评测 |
| **dickens, mozilla, webster, nci, xml, samba** | 多样化文本/二进制数据，用于泛化性验证 |

> 注：FASTQ 数据经过修正，避免质量字符串退化问题，保证实验有效性。

### ⚙️ 实验设置

| 组件 | 配置 |
|------|------|
| **硬件平台** | NVIDIA H100 80GB HBM3，132 SMs，CUDA 12.4 |
| **解码模式** | device-resident（压缩数据驻留 GPU 显存） |
| **pipeline 架构** | tile-ANS：熵解码 + 匹配执行两阶段分离 |
| **match kernel** | `k_decode_g`，cooperation width $ G \in \{8,16,32\} $，默认 $ G=32 $ |
| **block size 控制** | 通过 `ACEAPEX_BS` 编码参数设定（如 16384） |
| **timer scope** | **仅测量 match phase**，host-device transfer 与 CPU entropy 不计入时间 |
| **正确性验证** | GPU 路径用 FNV hash，CPU 路径用 byte compare，每一点都 bit-perfect |

### 📈 评估指标

| 指标 | 说明 |
|------|------|
| **Decode Throughput (GB/s)** | 主要性能指标，衡量 match phase 的吞吐能力 |
| **Compression Ratio** | 输出大小 / 输入大小，越高越好 |
| **Effective Workload (lanes)** | 定义为 $ \text{lanes} = G \times N_{\text{blocks}} $，反映并发粒度 |
| **Average Match Length** | 影响 work granularity 的关键变量 |

### 🔁 基线方法对比

- **Base Configuration**：原始 min-match-length 设置（6/8/10/12）
- **Tuned Configuration**：调整为（12/16/24/32）
- **无其他 decoder 对比**：因 focus 在机制分析而非端到端竞争，但强调与 CODAG、nvCOMP、Blackwell 引擎的区别

> ❗ 特别指出：ACEAPEX 的独特价值是三者共存：
> - compressed-residency
> - absolute-offset parallel decode
> - sub-millisecond region seek  
> 此组合目前仅见于该设计。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Tables III & IV）

#### 表：压缩率提升（Table III）

| Dataset | Base Ratio | Tuned Ratio | Change |
|--------|------------|-------------|--------|
| FASTQ 1GB | 3.90 | 3.97 | **+1.8%** |
| enwik9 | 2.64 | 2.77 | **+4.9%** |
| dickens | 2.58 | 2.71 | **+5.0%** |
| mozilla | 2.62 | 2.68 | **+2.3%** |
| webster | 3.09 | 3.23 | **+4.5%** |
| nci | 9.92 | 10.25 | **+3.3%** |
| xml | 6.29 | 6.70 | **+6.5%** |
| samba | 3.92 | 4.13 | **+5.4%** |

> ✅ 所有 8 个数据集压缩率全部提升，无一例外。

#### 表：解码吞吐量提升（Table IV）

| Dataset | Base (GB/s) | Tuned (GB/s) | Change |
|--------|--------------|---------------|--------|
| FASTQ 1GB | 142.6 | 178.6 | **+25.2%** |
| enwik9 | 91.6 | 163.5 | **+78%** |
| dickens | 17.6 | 25.5 | **+45%** |
| mozilla | 28.9 | 29.3 | **+1.4%** |
| webster | 47.1 | 54.8 | **+16%** |
| nci | 47.2 | 49.1 | **+4%** |
| xml | 7.6 | 9.1 | **+20%** |
| samba | 15.8 | 23.9 | **+51%** |

> ✅ 所有数据集 decode throughput 均显著提升，**enwik9 接近翻倍**。

### 🔬 消融实验结果

| 假设 | 实验方法 | 结果 | 是否成立 |
|------|---------|------|----------|
| Compute bound? | 使用纯 copy kernel (`purecopy.cu`) 测试 | 吞吐接近原 kernel（差 <4%） | ❌ 否 |
| Occupancy limited? | 强制增加 resident blocks（`launch_bounds`） | 因寄存器溢出导致性能下降（-9% ~ -14%） | ❌ 否 |
| Address scatter? | 对比 sorted vs scattered 地址 | 吞吐相同 | ❌ 否 |
| Launch parallelism? | 并发运行两个 decode stream | 总吞吐仅为单流的 ~70% | ❌ 否 |

✅ 最终唯一解释：**work granularity（平均 match length）**

#### 合成实验：Throughput vs Average Match Length（Table II）

| avg len (bytes) | 32 | 64 | 128 | 256 | 512 | 1024 |
|------------------|-----|-----|------|-------|-------|--------|
| Throughput (GB/s) | 212 | 416 | 607 | 692 | 734 | **744** |

> 📉 从 32B 到 1024B，吞吐提升 **3.5×**（212 → 744 GB/s）

而真实数据中：
- enwik9：平均 match length = **6.5 bytes**
- FASTQ：平均 match length = **10.1 bytes**
- 95%+ 的匹配小于 32 字节 → 处于曲线最陡低效区

---

## 4. 关键结论和发现

### 🧩 主要发现

1. **Decode throughput 的决定因素是 work granularity（平均 match length）**，而非传统认为的 compute、occupancy 或 memory bandwidth。
2. **短匹配严重浪费 GPU 并行资源**：一个 32-byte 匹配在 32-thread warp 中仅占用 1 thread，其余 idle。
3. **提高 min-match-length 可实现双赢**：
   - 移除远距离短匹配可降低 offset 编码带来的熵代价；
   - 返回的 literal 增加被 offset 节省抵消，总体压缩率反而上升；
   - 同时提升平均 match length，改善 warp 利用率，提高 throughput。
4. **所有增益完全来自 encode-side 调整**，decode kernel 无需任何变更，极具实用性。

> 💡 “One cause, two effects” —— 同一机制带来压缩率和性能双提升。

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **seek 粒度有限** | 当前为 read/block-level seek（read ID → block），尚未支持 chr:pos 坐标级随机访问 |
| **最优阈值数据依赖** | 将 min-match-length 进一步提升至 `16/24/32/48` 会降低 FASTQ 压缩率（3.97→3.93），表明存在边际最优 |
| **未突破硬件带宽上限** | enwik9 在调优后达到 ~217 GB/s plateau，多次 bypass 尝试失败，确认已达当前 granularity 下的物理极限 |
| **profiler 受限** | 云主机禁用 Nsight 计数器，occupancy 分析基于 CUDA API 计算而非实测 |
| **encode 速度慢** | 编码速度 0.3–3.4 GB/s，且数据依赖性强，本文不涉及 encode throughput 优化 |

### 🔮 未来工作方向

1. **实现坐标级随机访问（coordinate-level seek）**
2. **将机制整合进完整 device-resident pipeline**
3. **与生产级 decoder（如 nvCOMP、Gompresso）进行 head-to-head benchmark**
4. **探索自适应 min-match-length 策略，根据数据特征动态调节**
5. **扩展至其他 GPU 压缩格式中的 granularity 控制**

---

## ✅ 总结一句话

> 本文揭示了 **absolute-offset GPU LZ77 解码吞吐量的本质瓶颈是 work granularity（平均 match length）**，并通过一个简单的 **encode-time min-match-length 调整策略（12/16/24/32）**，在不改变 decode kernel 的前提下，**同时提升了压缩率和解码性能**，在所有测试数据上实现了无例外的双赢，为高性能 GPU 压缩系统的设计提供了新的理论基础与实用杠杆。

</details>

---

### 6. [Searching for Plans You Can Actually Build: A Realizability-Aware Full-Space Optimizer for MoE Training and Serving](https://arxiv.org/abs/2607.18631)

**Authors**: Quan Yuan, Jie Zhao  
**Category**: cs.DC  
**Published**: 2026-07-22  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.18631v1  

#### Abstract
Mixture-of-Experts (MoE) systems split a program's plan space in two: the space a cost model can rank, and the smaller space a real toolchain can actually build. Automatic optimizers rank the first and silently assume the two coincide -- so they can return a plan that is optimal on paper and impossi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Searching for Plans You Can Actually Build: A Realizability-Aware Full-Space Optimizer for MoE Training and Serving*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前主流的 MoE（Mixture-of-Experts）自动优化器在搜索最优训练或推理计划时，仅依赖**成本模型**（cost model）对计划空间进行排名，却**隐含假设该空间与实际工具链可构建的空间完全一致**。然而，现实是许多被成本模型评为“最优”的计划（如 `tp1/ppl/ep8` + `DualPipe`）因工具链限制（如缺少特定调度器、整除约束、sharding语义不支持等）而**根本无法生成可运行代码**。

这导致了一个系统性鸿沟：**成本模型能评分的空间 ≠ 工具链能构建的空间**。这种“纸上最优、实际不可行”的问题不是精度校准能解决的，而是架构层面的**类别错误**（category error）。

### 提出了什么新方法或新思路
作者提出 **MOEFS**（Mixture-of-Experts Full-Space Search），一个**实现性感知**（realizability-aware）的全空间优化器，其核心创新在于将“部署可行性”作为搜索过程中的**一等公民约束**（first-class search constraint），而非事后过滤。

#### 主要创新点：
- **Realizability Predicate（实现性谓词）**  
  复用代码生成器（emitter）本身作为“干运行”（dry run）来判断一个计划是否可构建。只要生成器抛出异常（如 `NotImplementedError`），即判定为不可构建。这确保了“唯一事实来源”（one source of truth），避免并行维护规则库。
  
- **Price, Don’t Prohibit（定价而非禁止）**  
  对于那些可以构建但带来额外开销的计划（如内核打补丁、通信重叠冲突），不直接禁止，而是通过测量其“实现税”（realization tax）——以乘法因子形式计入成本模型，让搜索器自行权衡代价。

- **Closed-Loop Full-Space Search（闭环全空间搜索）**  
  在单一计划表示下，联合搜索三个层级：
  - **并行策略**（parallelism: tp/pp/ep）
  - **通信调度**（schedule: 如 1f1b, interleaved, dualpipe）
  - **内核结构**（kernels: 融合边界、tile形状等）
  并统一输出 **Megatron 训练栈** 和 **SGLang 推理栈** 的启动命令。

- **预注册与哈希锁定评估**（Pre-registered, artifact-hashed evaluation）  
  所有预测结果在执行前写入冻结的、哈希锁定的裁决文件中，确保结果不可篡改，增强可信度。

### 相比现有方法的优势
| 维度 | 现有方法 | MOEFS |
|------|--------|-------|
| 构建可行性 | 忽略或事后检查 | 搜索过程中强制约束 |
| 成本模型准确性 | 依赖跨硬件外推 | 支持域限定校准（domain-scoped calibration） |
| 内核生成 | 固定或独立优化 | 与策略联合搜索 |
| 训练/服务统一 | 分离栈 | 单一计划生成双栈 |
| 结果可信度 | 报告选择性结果 | 预注册 + 全部如实报告（含失败） |

---

## 2. 核心实验方法和设置

### 实验平台
- **本地节点**：2×RTX4090（sm_89 架构）
- **远程节点**：8×H800（sm_90 架构，NVLink 全连接）

### 模型与任务
- 使用参考 MoE 模型（类似 DeepSeek-V3/Kimi K2 风格）
- 同时评估 **训练**（training）和 **服务**（serving）阶段的性能

### 评估指标
- **吞吐量**（Throughput）：tokens/s 或 tokens/s/GPU
- **成功率标准**：自动搜索方案 ≥ 最优手工调优方案 × 0.98
- **对比方式**：
  - 与最强的手工调优配置对比
  - 报告原始浮点数值（raw-float），不进行平滑处理
  - 多轮中位数（3-round median）为主，辅以确认轮次

### 基线方法
- **手工调优基线**（hand-tuned baseline）：
  - Megatron-LM v0.18.0 + SGLang 0.5.15
  - 包括不同并行度（tp/pp/ep）、调度策略（1f1b, interleaved）、是否启用 overlap 等组合
- **默认成本模型 vs. 校准后成本模型**：验证校准效果

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table II）

| 机器 × 阶段 | 自动方案 (Auto.) | 手工最优 (Hand-tuned) | 比值 / 结果 |
|------------|------------------|------------------------|-----------|
| 2×RTX4090 训练 | 38,028.3 tok/s | 37,703.4 tok/s | **+2.9% PASS** |
| 2×RTX4090 服务 | 3,510.9 tok/s | 3,549.9 tok/s | 0.9890 PASS |
| 8×H800 服务 | 12,274.6 tok/s | 11,912.5 tok/s | **1.0304 PASS** |
| 8×H800 训练 | 12,059.0 tok/s/GPU | 12,913.2 tok/s/GPU | **0.9338 FAIL** |

> ✅ **说明**：前三个为通过项，最后一项为明确失败项。

### 详细分析

#### ✅ 成功案例
- **2×RTX4090 训练**：超出最强基线 **+2.9%**，且超过接受门槛（0.98×）达 **+2.9%**
- **8×H800 服务**：达到 **1.0304×** 手工配置吞吐，实现轻微超越（实为同配置下的运行波动）

#### ❌ 失败案例（关键发现）
- **8×H800 训练**：最终方案为 **computable honest FAIL**，达到手工最优的 **93.38%**，未达 98% 门槛。
  - **原因**：仅在一个调度标志（overlap 是否开启）上失利。
  - **重要性**：该计划**可构建且可运行**，但性能不如手工方案，属于真实性能损失，非构建失败。

#### 实现性谓词的效果（RQ1）
- 在 8×H800 上，未经实现性约束的 top-5 训练计划全部**不可构建**（0/6 measurable）。
- 引入实现性谓词后，搜索到的 top-1 可构建方案相比“先搜再滤”的回退策略，**预测吞吐提升 +10.8%**。

#### 内核生成与实现税（RQ4）
- 在小专家数（me < 32）场景下，生成内核优于 DeepEP；
- 在大专家数（me > 32）时，DeepEP 更优（最高落后 51.7% @ me=1024）。
- **实现税跨代翻转**：
  - **服务侧**：RTX4090 上 +41.7%，H800 上高达 **+123.6%**（因 H800 原生 fused_moe 更快）
  - **训练侧**：RTX4090 上 +26.6%，H800 上变为 **-7.7%**（负税，即生成内核更快）

#### 消融实验（RQ5）
- 在本地小网格上，各层级（L1/L2/L3）搜索结果一致，表明**平坦性源于测试规模不足**。
- 在更大模型（kimi_k2）上，修正内存账目后，L1 到 L2 的吞吐提升 **+8.11% → +8.51%**，证明搜索具有判别力。

---

## 4. 关键结论和发现

### 主要发现
1. **构建可行性是独立维度**  
   成本模型即使完美校准，仍可能推荐无法构建的计划。**实现性 ≠ 准确性**，必须作为显式约束引入搜索。

2. **闭环设计暴露隐藏问题**  
   只有当搜索、生成、部署形成闭环时，才能观察到“最优计划不可构建”的现象。开放流程无法发现此问题。

3. **“定价而非禁止”更灵活**  
   对实现开销进行量化而非一刀切禁止，保留了潜在收益方案（如负税情况），提升了搜索灵活性。

4. **预注册增强可信度**  
   所有预测在运行前冻结，后续结果无论成败均如实报告，包括多次自我证伪（如 v3c 假设被 v3d 推翻），体现了科学严谨性。

5. **成功与失败并存**  
   - 在 **2×RTX4090 训练** 和 **8×H800 服务** 上达成目标；
   - 在 **8×H800 训练** 上明确失败，但失败本身是有价值的结果。

### 方法的局限性
| 局限性 | 描述 |
|--------|------|
| 单节点评估 | 未涉及多节点拓扑感知通信（如 topology-aware collectives） |
| 小模型规模 | 前沿大模型在单节点上内存不可行（HN19） |
| 发布版本绑定 | 实现性判断基于 Megatron v0.18.0 和 SGLang 0.5.15，升级后需重新验证 |
| Launch Probe 未自动化 | 当前 launch probe 是手动流程，未集成进 `search()` |
| 校准未跨工作负载 | 服务端校准在同一模型上完成，缺乏 holdout 验证 |

### 未来工作方向
1. **将 launch probe 集成进搜索循环**，实现全自动闭环。
2. **支持 CUDA-C++ 内核生成后端**，突破 Triton 当前限制。
3. **建模通信模式不对称性**（如 all-to-all vs all-gather）带来的残余收益。
4. **探索生成内核何时应让位于手调库**（如 DeepEP），建立切换策略。
5. **扩展至多节点、大规模集群环境**，验证实现性谓词的可扩展性。

---

> **总结一句话**：  
> MOEFS 首次将“计划是否可构建”从后处理变成搜索的一等公民约束，通过 **emit dry-run + 定价机制 + 预注册评估**，实现了对 MoE 训练与服务的**真实可行**的全空间优化，并坦率报告了成功与失败，推动了系统研究的透明化与可复现性。

</details>

---

### 7. [InstantInfer: Enabling Fast LLM Cold Start with Communicating Finite Automata](https://arxiv.org/abs/2607.18957)

**Authors**: Yitao Yuan (Peking University, Scitix AI), Yongchao He (Scitix AI), Shaoke Fang (Peking University), Wenfei Wu (Peking University)  
**Category**: cs.DC  
**Published**: 2026-07-22  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.18957v1  

#### Abstract
Cold starts in large language model (LLM) inference services significantly affect user experience, yet they remain inefficient due to sequential initialization and a massive number of fine-grained I/O requests issued by complex software components. Although refactoring the program can yield advantag...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：InstantInfer: Enabling Fast LLM Cold Start with Communicating Finite Automata**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
大型语言模型（LLM）推理服务中的**冷启动延迟**（cold start latency）严重影响用户体验。在弹性部署、突发请求、模型切换等场景下，系统需要从磁盘加载模型到 GPU 并完成初始化，这一过程耗时极长（可达数百秒），占 **Time to First Token (TTFT)** 的 99.6% 以上。

传统方法存在两大瓶颈：
- **串行执行**：组件（如进程、张量）按顺序初始化，无法利用现代硬件的并发能力。
- **细粒度 I/O 开销**：张量加载以单个 tensor 为单位，导致大量小规模 I/O 请求，难以匹配底层存储系统的高效块操作粒度。

---

### **提出的新方法与新思路**
论文提出了 **Communicating Finite Automata (CFA)** 抽象和编程框架，用于系统性重构 LLM 冷启动程序：

#### **核心创新点**：
1. **CFA 抽象建模**  
   将每个初始化组件（如进程、张量、模型实例）抽象为一个有限状态机（Finite Automaton, FA），其状态单调递增（如 `Start → Constructed → Ready`）。跨组件依赖通过显式的**状态依赖关系**（state dependency）表达，例如 `(Parent, Ready) → (Child, Constructed)`。

2. **安全并发与 I/O 合并优化**
   - **并发执行**：无依赖的状态转换可安全并行执行，打破原有串行控制流。
   - **I/O 合并**：多个细粒度 FA 可合并为粗粒度 FA，实现批量 I/O 操作，提升硬件带宽利用率。

3. **轻量级编程框架**
   提供统一的声明式接口 `channel.set_state()` 和 `channel.wait_state()`，开发者只需少量插入语句即可启用 CFA 重构，保留原代码结构，降低改造成本。

4. **形式化正确性证明**
   证明了 CFA 重构后的并发执行与原始串行执行在最终状态上是等价的，确保功能正确性。

---

### **相比现有方法的优势**
| 维度 | 现有方法 | InstantInfer |
|------|--------|-------------|
| **优化粒度** | 单阶段优化（如仅加速加载或编译） | 跨组件、全流程协同优化 |
| **并发支持** | 手动改写易出错 | 安全自动并发，基于状态依赖 |
| **I/O 效率** | 张量级细粒度读取 | 块（chunk）级聚合 I/O，支持 AllGather 流水线 |
| **通用性** | 特定场景定制 | 可应用于进程树构建、张量加载、模型切换等多个场景 |
| **集成难度** | 需深度重构 | 增量式集成，修改约 3000 行 Python + 2700 行 C++ |

---

## **2. 核心实验方法和设置**

### **数据集与模型**
使用主流开源 LLM 进行测试，涵盖不同规模和架构：
- **中小模型**：GPT-2, OPT-13B, Llama-2-7B/70B
- **大模型**：Llama-3.1-70B/405B, Qwen3-30B-A3B/235B-A22B, DeepSeek-R1 (~400B)
- 包含 Dense 和 MoE 架构，支持 Tensor Parallelism (TP) 和 Expert Parallelism (EP)

---

### **实验设置**
#### **硬件平台**
- **H20 Testbed**：8× NVIDIA H20 GPU (141GB)，NVLink 互联，50 GB/s GPFS 存储
- **L40 Testbed**：8× NVIDIA L40 GPU (48GB)，PCIe 互联，20 GB/s 存储
- 每节点配 2TB 主机内存，Intel Xeon Platinum 8468 CPU（192 核）

#### **基线方法对比**
| 基线 | 类型 | 说明 |
|------|------|------|
| **vLLM (0.13.0)** | 主流推理框架 | 默认配置，代表传统非 serverless 场景 |
| **SGLang (0.5.9)** | 结构化推理引擎 | 支持动态图，但冷启动慢于 vLLM |
| **ServerlessLLM (SLLM, 0.8.0)** | Serverless 专用系统 | 显式优化冷启动，当前最优之一 |
| **Standalone Loaders** | 加载器对比 | Safetensors, fastsafetensors, Run:ai Model Streamer |

所有系统均启用 `torch.compile` 缓存，并从存储而非内存加载模型以模拟真实冷启动。

---

### **评估指标**
| 指标 | 描述 |
|------|------|
| **TTFT (Time to First Token)** | 用户感知延迟，主指标 |
| **Engine Startup Time** | 引擎初始化时间 |
| **Model Loading Time** | 模型从磁盘加载至 GPU 时间 |
| **Service Stall Time** | 模型切换期间不可服务的时间 |
| **Hardware Utilization** | 存储吞吐、GPU 利用率 |
| **CPU/Memory Overhead** | 资源开销分析 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### **端到端冷启动 TTFT**
- 在 H20 上，InstantInfer 相比：
  - **ServerlessLLM**：快 **2.7×–2.9×**
  - **vLLM/SGLang**：快 **3.2×–7.2×**
- 在 L40 上：
  - **ServerlessLLM**：快 **1.8×–2.6×**
  - **vLLM/SGLang**：快 **2.1×–3.7×**

> 图 7 显示 CDF 曲线显著左移，表明延迟分布全面改善。

#### **多实例并发冷启动扩展性**
- 启动 8 个并发实例时，InstantInfer 仍保持 **1.6×–2.3×** 优势。
- 随着实例数增加，延迟增长主要来自进程间资源竞争（如包导入、CUDA 初始化），但 CFA 并发有效缓解了该问题。

#### **模型加载速度**
- **从存储加载**（H20）：
  - 比 Safetensors 快 **10.4×–32.3×**
  - 比 fastsafetensors 快 **3.3×–9.0×**
  - 比 Run:ai 快 **4.8×–7.0×**
  - 比 ServerlessLLM 快 **1.8×–6.5×**
- **从内存加载**：
  - 仍比 Safetensors 快 **1.9×–10.1×**
  - 在 PCIe 环境（L40）中也保持 **1.3×–3.1×** 优势

> 性能提升源于 CFA 构建的高度重叠的 chunk I/O 流水线，实现了接近理论带宽的存储吞吐（达链路容量的 80%）。

#### **模型切换延迟**
- **服务中断时间（Service Stall Time）**：
  - 比 ServerlessLLM 减少 **3.9×–5.0×**
  - 比 vLLM/SGLang 减少 **4.6×–11.8×**

> 通过将旧模型释放 GPU 内存与新模型初始化重叠，极大缩短停服窗口。

---

### **消融实验结果**
#### **各模块贡献分解**（图 11）
- **Tensor Materialization**：带来 **2.4×–3.7×** 加速（主导因素）
- **Process-Tree Materialization**：额外 **1.2×–1.5×** 加速
- **Model Switching Overlap**：再提速 **1.6×–2.1×**
- **总体复合加速**：**5.0×–7.8×** 超过未优化基线

#### **CPU 与内存开销**
- **CPU 使用**：约 **2.5–3.5 核/GPU**，高于 vLLM（~1 核/GPU），但在百核服务器中可接受。
- **内存占用**：临时缓冲区在加载完成后立即释放，不影响运行时 KV Cache。
- **兼容性优势**：无需转换模型格式，而 ServerlessLLM 需为每种并行配置保存副本。

---

## **4. 关键结论和发现**

### **主要发现**
1. **冷启动瓶颈本质是软件结构问题**  
   不是硬件限制，而是串行控制流和细粒度 I/O 导致的低效。通过 CFA 抽象可系统性暴露优化机会。

2. **CFA 实现安全高效的并发与聚合**
   - 状态依赖机制保障并发安全性
   - chunk 抽象实现 I/O 合并与流水线，最大化硬件利用率

3. **InstantInfer 具备强鲁棒性和可扩展性**
   - 在不同 GPU（H20/NVLink vs L40/PCIe）、模型规模（<10B 到 >400B）、负载模式下均表现优异
   - 支持 burst 请求场景，快速响应流量激增

4. **与现有优化正交且可组合**
   - 可与 Medusa（CUDA graph 复用）、HydraServe（流水线预热）等技术结合，进一步压缩冷启动路径

---

### **局限性**
1. **对高度耦合组件支持有限**  
   若组件间存在复杂共享状态或反向依赖，CFA 建模可能变得复杂。

2. **初始映射需人工设计**  
   chunk 与 tensor 的映射关系需预计算，虽可自动化但仍增加部署复杂度。

3. **侧重初始化阶段**  
   未优化 `Optimize` 阶段（如 graph capture），需与其他技术配合才能完全消除冷启动影响。

---

### **未来工作方向**
1. **自动化 CFA 建模工具链**
   开发静态分析工具，自动识别组件边界与状态依赖，降低人工干预。

2. **动态自适应调度**
   根据实时资源状况动态调整 chunk 分片策略与并发度。

3. **与预热机制融合**
   探索“部分冷启动 + CFA 加速”的混合模式，在资源受限环境下取得更好平衡。

4. **扩展至其他系统场景**
   将 CFA 抽象推广至训练启动、分布式作业调度等同样存在初始化瓶颈的系统中。

--- 

> ✅ **总结一句话**：  
> InstantInfer 通过 **Communicating Finite Automata** 抽象，首次实现了对 LLM 冷启动过程中**跨组件并发执行**与**细粒度 I/O 合并**的安全、系统性优化，在多种硬件与模型上实现最高 **7.2× 的 TTFT 加速**，为高性能、高弹性的 LLM 服务提供了新的基础设施范式。

</details>

---

### 8. [Fishing Out Free Riders: Shapley-Based Reward Attribution for Parallel Reasoning via Reinforcement Learning](https://arxiv.org/abs/2607.18979)

**Authors**: Wentao Zhang, Haoyu Zhang, Xinke Jiang, Yuxuan Cheng, Yuhan Pan, Miao Li, Zhipeng Qiao, Tao Feng, Zhen Tao, Dengji Zhao  
**Category**: cs.AI  
**Published**: 2026-07-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.18979v1  

#### Abstract
Large Language Models (LLMs) excel at multi-step reasoning, yet current parallel reasoning approaches often fail to distinguish the contributions of individual reasoning paths. Many paths may be redundant, misleading, or even detrimental, but outcome-level rewards assign uniform reward, leading to a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Fishing Out Free Riders: Shapley-Based Reward Attribution for Parallel Reasoning via Reinforcement Learning*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前的 **Parallel Reasoning**（并行推理）方法在训练过程中通常采用 **outcome-level reward**（基于最终答案正确性的奖励），将相同的奖励均匀分配给所有生成的推理路径（reasoning paths）。这种机制存在严重缺陷：
- **无法区分路径贡献**：冗余、误导甚至错误的路径可能因最终答案正确而获得正向梯度更新，形成“搭便车”（free rider）现象。
- **奖励信号模糊**：由于多个路径被汇总为一个 `<Summary>`，其过程级贡献难以追溯，导致学习信号不明确，训练不稳定。

### 提出了什么新方法或新思路
作者提出 **Parallel Shapley**，一种基于 **Shapley Value** 的细粒度奖励归因框架，用于多路径并行推理的强化学习训练。核心思想包括：

- **将每条推理路径视为合作博弈中的“玩家”**，利用 **Shapley Value** 量化其对最终结果的边际贡献。
- 设计 **两层级奖励机制**：
  - **Outcome-level reward**：基于最终答案正确性（如字符串匹配）。
  - **Process-level reward**：通过 **Generative Reward Model (GRM)** 和 **Monte Carlo Sampling** 近似计算每条路径的 Shapley 值作为路径级奖励。
- 在 **Group Relative Policy Optimization (GRPO)** 框架下进行训练，实现 token-level 的策略梯度更新。

### 相比现有方法的优势
- ✅ **更精细的奖励分配**：避免“搭便车”，有效抑制误导性和冗余路径。
- ✅ **更强的训练稳定性与更快收敛**：提供更密集、更具判别力的学习信号。
- ✅ **提升推理多样性与互补性**：鼓励模型生成非重复、互补的推理路径，提高解空间覆盖能力。
- ✅ **可解释性强**：Shapley 值提供了路径贡献的可解释性分析工具。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Parallel-GSM8K**：用于监督微调（SFT）冷启动阶段，包含约 7,473 条带并行推理轨迹的小学数学题。
- **DAPO**：用于 RL 训练的大规模数学推理数据集，含 17,000 个带整数答案的问题。
- **测试集**：
  - **AIME25 / AIME24**：奥赛级别数学推理基准。
  - **AMC23**：竞赛级数学推理基准。
  - **MATH**：涵盖七个学科领域的高难度数学问题集。

### 实验设置
- **模型架构**：以 **Qwen3-4B-Base** 为主干模型。
- **训练流程**：
  1. **SFT 冷启动**：在 Parallel-GSM8K 上训练 230 步。
  2. **RL 微调**：在 DAPO 上使用 GRPO 算法训练 40 步，batch size=256，learning rate=1e-6。
- **路径数量（K）**：默认 K=4，并进行了消融研究（K=2~5）。
- **GRM**：默认使用 `qwen-plus` 作为生成式奖励模型，评估路径子集的效用。

### 评估指标
- **Mean@16**：16 次独立采样下的平均准确率。
- **Pass@16**：至少有一次采样正确的比例（反映模型潜力上限）。
- **MATH 数据集使用 Mean@1**（单次采样）。

### 基线方法对比
| 方法 | 类型 | 描述 |
|------|------|------|
| **Qwen3-4B-Base** | 基线 | 未经并行推理训练的基础模型 |
| **Parallel-SFT-Unseen** | SFT | 仅使用 SFT 训练的并行推理模型 |
| **GRPO (DAPO)** | RL | 使用 GRPO 的标准 outcome-level reward 方法 |
| **Parallel-R1-Unseen** | RL | 当前 SOTA 并行推理模型，使用 outcome-level reward |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 1）
| 方法 | AIME25 Pass@16 | AIME24 Pass@16 | AMC23 Pass@16 | MATH Mean@1 | **Avg. Pass@16** |
|------|----------------|----------------|---------------|-------------|------------------|
| Parallel-R1-Unseen | 37.8 | 33.2 | 88.9 | 82.6 | 47.1 |
| **Parallel Shapley** | **50.0** (+32.3%) | **63.3** (+90.7%) | **92.5** (+4.0%) | **83.1** | **48.3** |

- **平均 Pass@16 提升 42.3%**，显著优于所有基线。
- 在最难的 **AIME25** 上，Pass@16 从 37.8 提升至 **50.0**，绝对增益达 **12.2 个百分点**。
- **仅用 40 步 RL 训练**，性能已超越训练 200 步的 Parallel-R1。

### 与基线方法的对比结果
- **相比 Parallel-R1**：
  - 更快收敛（见图3），训练约 30 步即稳定，而 Parallel-R1 持续缓慢上升。
  - 显著提升 **Pass@16**，说明模型探索出更多正确解路径。
  - **Mean@16 略有下降或持平**，表明模型更倾向于多样化探索而非单一最优路径（合理权衡）。
- **相比 GRPO/SFT**：全面大幅领先，验证了并行推理 + 细粒度奖励的有效性。

### 消融实验结果（Ablation Study）

#### （1）路径控制策略对比（Table 1）
| 方法 | Avg. Pass@16 |
|------|-------------|
| **Parallel Shapley** | **48.3** |
| Leave-One-Out (LOO) | 46.2 |
| Independent Path Evaluation | 46.5 |

- **LOO** 仅衡量全集下的边际影响，未考虑不同子集组合。
- **Independent Evaluation** 忽略路径间交互，效果最差。
- 表明 **Shapley 的加权平均机制** 对建模路径交互至关重要。

#### （2）路径数量 K 的敏感性（Table 2）
| K | AIME25 Pass@16 | AIME24 Pass@16 | AMC23 Pass@16 | Avg. Pass@16 |
|---|----------------|----------------|---------------|--------------|
| 2 | 16.0 | 18.1 | 67.2 | 45.4 |
| 3 | 16.4 | 17.0 | 67.8 | 45.6 |
| **4** | **17.2** | **20.1** | **72.8** | **48.3** |
| 5 | 15.2 | 18.9 | 69.4 | 46.2 |

- **K=4 时达到最优性能**，进一步增加路径数反而降低表现，说明存在“收益递减”效应。

#### （3）GRM 敏感性分析
- 即使使用较弱的开源 GRM（如 Qwen2.5-3B-Instruct），**Parallel Shapley 仍优于 Parallel-R1**。
- GRM 越强，性能越好，说明 **Shapley 归因与 GRM 质量正相关**。

#### （4）路径互补性分析（Figure 4）
- 在 **20% 和 40% 路径 dropout** 下，Parallel Shapley 的性能下降幅度远大于 Parallel-R1。
- 说明其路径之间 **高度互补、信息非冗余**，移除任一路径都会造成显著损失。

---

## 4. 关键结论和发现

### 主要发现
1. **Shapley 值能有效“钓出搭便车者”**：通过量化路径的边际贡献，避免误导路径获得不当奖励。
2. **细粒度奖励显著提升训练效率与性能**：仅 40 步 RL 即超越 SOTA，且训练更稳定。
3. **模型学会生成互补路径**：不同路径承担不同推理角色（如代数变换、边界验证等），形成“分工协作”。
4. **Pass@16 与 Mean@16 存在权衡**：Shapley 鼓励多样化探索，提升了解的覆盖率（Pass@16），但单次生成质量（Mean@16）略有牺牲。

### 方法的局限性
1. **计算开销较大**：Monte Carlo 估计需多次调用 GRM，每步训练时间比 Parallel-R1 多约 62%（见 Table 4）。
2. **尚未在更大模型上验证**：实验基于 Qwen3-4B-Base，未扩展到更大规模 LLM。
3. **Pass@16 与 Mean@16 的差距仍存**：如何将多路径优势“内化”为单次生成能力仍是挑战。
4. **GRM 查询成本随 K 增长**：路径数较多时可能成为瓶颈。

### 未来工作方向
1. **On-policy Self-Distillation**：利用多样化的多路径输出作为蒸馏目标，提升单次生成性能。
2. **更高效的 Shapley 近似算法**：如分层采样（stratified sampling）以减少 GRM 查询次数。
3. **扩展到其他任务**：如代码生成（模块分解）、multi-hop QA（证据链构建）等。
4. **在更大模型上验证与部署**。

---

> **总结**：*Parallel Shapley* 是首个将 **cooperative game theory** 引入并行推理训练的工作，通过 **Shapley-based reward attribution** 实现了对推理路径的细粒度建模，有效解决了“搭便车”问题，在数学推理等多个任务上实现了显著且稳定的性能提升，为高质量多路径推理提供了新的范式。

</details>

---

### 9. [HPD-Parsing: Hierarchical Parallel Document Parsing](https://arxiv.org/abs/2607.18839)

**Authors**: Shu Wei, Jingjing Wu, Lingshu Zhang, Qunyi Xie, Hao Zou, Le Xiang, Xu Fan, Yangliu Xu, Manhui Lin, Xiaolong Ma, Cheng Cui, Tengyu Du,  YY  
**Category**: cs.CL  
**Published**: 2026-07-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.18839v1  

#### Abstract
Efficient teamwork typically combines global coordination with parallel execution, a principle not yet fully reflected in unified Vision-Language Model (VLM)-based document parsers. Existing unified parsers process an entire page jointly but generate its output through a single token-by-token autore...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《HPD-Parsing: Hierarchical Parallel Document Parsing》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的基于 **Vision-Language Model (VLM)** 的统一文档解析器普遍采用全页 **autoregressive decoding**（自回归解码），即逐 token 生成整个页面的输出。这种序列化生成方式导致推理延迟高、吞吐量低，尤其在处理长文本或复杂布局时形成显著的**顺序瓶颈**（sequential bottleneck）。尽管已有研究尝试通过压缩视觉 token 或引入多 token 预测来优化效率，但仍未从根本上重构文档生成的解码范式。

### 提出的新方法与新思路
本文提出 **HPD-Parsing**，引入一种全新的 **Hierarchical Parallel Decoding**（分层并行解码）范式，其核心思想是：
- **全局协调 + 局部并行**：文档结构（layout）需全局分析，而各区块内容可独立并行解析。
- 构建一个主干的 **main layout branch** 负责解析整体布局和阅读顺序，并动态触发多个并发的 **content branches** 来并行解码各个局部区域的内容。
- 在每个分支内部集成 **Progressive Multi-Token Prediction (P-MTP)** 技术，进一步减少每步所需的 autoregressive 步数。
- 所有分支共享前缀的 **KV cache**，避免重复计算，提升效率。

### 相比现有方法的优势
- ✅ **打破全页自回归限制**：不再依赖单一的 token-by-token 全局轨迹，显著缩短有效解码路径。
- ✅ **高吞吐、低延迟**：支持跨区块并发解码 + 分支内多 token 预测，大幅提高 TPS 和 PPS。
- ✅ **保持高精度**：通过 staged adaptation 和难度感知的数据构建策略，在加速的同时维持 competitive parsing accuracy。
- ✅ **统一框架下的高效性**：相比 pipeline 方法更少误差传播，相比传统 unified VLM 更高效。

---

## 2. 核心实验方法和设置

### 使用的数据集
- 主要评测基准：**OmniDocBench v1.6**
  - 包含多样化的文档类型（学术论文、表格、公式、新闻等）
  - 综合评估文本识别、公式识别、表格结构提取、阅读顺序预测等多个维度

### 实验设置
- **模型架构**：
  - Backbone：`InternVL3.5-1B`（0.3B 视觉编码器 + 0.8B LLM 解码器）
  - 解码器基于 `Qwen3-0.6B` 修改，使用 **Grouped-Query Attention (GQA)** 和 **SwiGLU** 激活函数
- **训练策略**：三阶段 staged adaptation
  1. **Stage 1**: 使用全页 autoregressive 格式初始化模型能力（学习通用解析能力）
  2. **Stage 2**: 切换至 hierarchical 并行格式，进行解码范式迁移 + 困难样本优化
  3. **Stage 3**: 引入轻量级 **Reinforcement Learning** 进行 reward-guided 优化（如公式质量、表格一致性等）
- **推理系统**：基于 **vLLM 0.17.1** 实现，扩展其调度机制以支持动态分支 fork 和 KV cache 共享
- **硬件平台**：8×NVIDIA A800 GPU（80GB 显存）

### 评估指标
| 指标 | 含义 |
|------|------|
| **TextEdit ↓** | 文本编辑距离（越小越好） |
| **Formula CDMT ↑** | 公式识别准确率 |
| **Table TEDS ↑ / TEDS-S ↑** | 表格结构匹配度（含/不含内容） |
| **ReadOrderEdit ↓** | 阅读顺序错误率 |
| **Overall Score** | 加权平均得分 |
| **TPS**（Tokens Per Second） | 每秒处理 token 数（衡量吞吐） |
| **PPS**（Pages Per Second） | 每秒处理页数 |

### 基线方法对比
- **Pipeline 类**：
  - `PaddleOCR-VL`, `MinerU-2.5-Pro`, `Youtu-Parsing`
- **Unified 类**：
  - `DeepSeek-OCR-2`, `Unlimited OCR`, `HunyuanOCR-1.5`, `Qianfan-OCR`, `FireRed-OCR`, `GLM-OCR`
- **Autoregressive Baseline**：
  - 同 backbone 下的标准 autoregressive 版本作为对照

---

## 3. 主要实验结果和性能指标

### 关键性能数据（OmniDocBench v1.6）

| 模型 | 参数量 | Overall Score | TPS (BS=512) | PPS (BS=512) |
|------|--------|----------------|---------------|---------------|
| Autoregressive Baseline | 1B | — | 1,554.8 | 1.02 |
| **HPD-Parsing (Ours)** | **1B** | **94.91** | **4,752.1** | **2.68** |

> 🔺 **吞吐提升**：
- 较 autoregressive baseline：**TPS 提升 3.06×**，**PPS 提升 2.62×**
- 较当前最快 unified 模型（DeepSeek-OCR-2）：**TPS 提升 1.62×**

### 与基线方法的对比结果
- **精度方面**：
  - 以仅 1B 参数超越多数更大模型（如 4B 的 Qianfan-OCR、Logics-Parsing-v2）
  - 在 Formula CDMT 上达 **97.28**，Table TEDS-S 达 **94.11**，表现 SOTA
  - ReadOrderEdit 为 **0.124**，优于大多数 unified 模型
- **效率方面**：
  - 即使输入 token 数高达 ~4,800（约为 DeepSeek-OCR-2 的 4 倍），仍实现更高吞吐
  - 输出越长，优势越明显（见下图）

#### 效率随输出长度的增长趋势（Figure 5）
| 输出长度区间 | 解码步数减少 | 请求吞吐提升 | 单请求延迟降低 |
|-------------|--------------|----------------|--------------------|
| 最长桶（>7k tokens） | **18.04×** | **3.67×** | **5.80×** |

> 💡 说明 HPD-Parsing 的加速效果在长文档中尤为显著，因其关键路径由最长分支决定，而非总长度累加。

### 消融实验结果（文中未直接列出表格，但从分析可推知）
- **消融 P-MTP**：会导致每分支解码步数上升，整体 TPS 下降约 30–40%
- **移除 shared-prefix KV cache reuse**：新增分支需重新编码前缀，带来显著 prefills 开销，影响并发效率
- **仅用 layout branch（退化为 autoregressive）**：吞吐回落至 baseline 水平
- **验证 staged adaptation 必要性**：跳过 Stage 1 直接训练 hierarchical 格式会导致收敛困难和精度下降

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **文档解析天然适合“全局协调 + 局部分解”**：
   - 布局必须全局建模，但内容可高度局部化，无需强序列依赖。
2. ✅ **Hierarchical Parallel Decoding 是有效的替代范式**：
   - 取代全页 autoregressive 生成，能同时提升效率与鲁棒性。
3. ✅ **高吞吐与高精度可以兼得**：
   - 通过 staged adaptation 和 difficulty-aware 数据构建，成功将在 autoregressive 上学到的能力迁移到并行解码中。
4. ✅ **错误隔离机制增强鲁棒性**：
   - 并行分支之间相互隔离，防止重复错误在整个文档中传播（如 Figure 9(c) 所示）。

### 方法的局限性
- 🚧 **对极复杂嵌套结构支持有限**：当前 fork 机制假设块间相对独立，对于深度嵌套（如表格内的公式再嵌套表格）可能需要递归扩展。
- 🚧 **依赖高质量 layout detection**：虽然比 pipeline 更鲁棒，但仍需 reasonably accurate bounding box 定位来划分 content branch。
- 🚧 **工程实现复杂度较高**：需定制推理引擎支持动态分支 fork、KV 共享与回收，部署门槛高于标准 VLM。

### 未来工作方向
- 🔮 探索 **multi-page document understanding** 中的 hierarchical decoding 扩展
- 🔮 发展 **structure-aware attention mechanisms**，进一步压缩 attention context
- 🔮 扩大数据覆盖范围，特别是更多语言、手写体、扫描质量差的文档
- 🔮 将该范式推广至其他结构化生成任务，如 **Key Information Extraction (KIE)**、**Document QA** 等

---

> ✅ **一句话总结**：  
> HPD-Parsing 通过提出 **Hierarchical Parallel Decoding** 新范式，实现了 **高吞吐（4,752 TPS）与高精度（Overall 94.91）的统一**，为下一代高效文档智能系统提供了新方向。

</details>

---

### 10. [Visual Semantic Decoding of Electrocorticography from Video Stimuli using End-to-End Deep Learning](https://arxiv.org/abs/2607.18923)

**Authors**: Stella Ho, Joel Villalobos, Joseph West, Jingyang Liu, Weijie Qi, Haruhiko Kishima, Ryohei Fukuma, Takufumi Yanagisawa, Sam E. John, David B. Grayden  
**Category**: cs.LG  
**Published**: 2026-07-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.18923v1  

#### Abstract
ECoG-based visual semantic decoding enables inference of semantic interpretation of visual perception from complex, noisy brain activity. This study examines the feasibility of visual semantic decoding using an end-to-end deep learning framework using electrocorticography (ECoG). Specifically, the d...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Visual Semantic Decoding of Electrocorticography from Video Stimuli using End-to-End Deep Learning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本研究聚焦于**视觉语义解码**（visual semantic decoding）任务，即从动态视频刺激下的**高噪声、复杂时序的ECoG信号**中直接推断出大脑对视觉内容的语义理解。该问题在临床脑机接口（BCI）中具有重要意义，但面临以下挑战：
- 数据稀缺（每个类别仅约50个训练样本）
- 个体间电极覆盖差异大
- 动态视频缺乏明确的刺激起始点
- 深度学习模型常被视为“黑箱”，可靠性受限

### 🚀 提出的新方法与创新思路
作者提出了一种**端到端深度学习框架**（end-to-end deep learning framework），无需手工特征工程即可实现从原始ECoG时间序列到视觉语义类别的映射，并系统评估了多种低资源学习策略。

主要创新点包括：
1. **系统比较三种低资源DL方法**：
   - **Few-shot Learning**：使用 Prototypical Networks 进行小样本分类
   - **Data Augmentation**：采用 Mixup 和 Manifold Mixup 增强数据多样性
   - **Self-supervised Learning**：利用 SimCLR 对大量无标签ECoG进行对比预训练
2. **引入Transformer架构用于ECoG解码**，并分析其在不同频带输入下的表现
3. **多维度可解释性分析**：从**频谱**（spectral）、**时间**（temporal）和**皮层区域**（cortical）三个维度深入剖析模型行为，验证其神经科学合理性
4. **优化后处理窗口**：通过分析发现更长的post-stimulus窗口（如900ms）优于传统500ms设定

### 🔍 相比现有方法的优势
| 维度 | 优势说明 |
|------|----------|
| **方法论** | 不依赖人工设计特征（handcrafted features），实现真正的端到端建模 |
| **实用性** | 在少于50个样本/类别的极端数据稀缺条件下仍取得良好性能 |
| **可解释性** | 模型决策机制与已知神经科学知识一致（如V1/V2、MT+区的作用），提升临床可信度 |
| **泛化能力** | 跨被试、跨会话的Leave-One-Recording-Out（LORO）协议增强了结果稳健性 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **来源**：Fukuma et al. (2022) 收集的ECoG视频观看数据集 [18]
- **参与者**：n = 17 名药物难治性癫痫患者（drug-resistant epilepsy）
- **记录方式**：subdural ECoG电极植入，采样率10kHz → 下采样至500Hz
- **任务范式**：观看6段各10分钟的电影预告片、幕后花絮等自然视频
- **目标类别**：三类视觉语义类别 —— `Human Face`、`Text`、`Landscape`
- **标注策略**：手动筛选每类50个高质量片段（共150个有标签样本），其余为无标签数据

### ⚙️ 实验设置
- **输入信号**：将连续ECoG切分为非重叠的时间窗（window）
- **频率滤波**：分别测试 alpha (8–13 Hz)、beta (13–30 Hz)、low-gamma (40–80 Hz)、high-gamma (80–150 Hz)、broadband-gamma (40–150 Hz) 和 broadband (3–250 Hz)
- **时间窗长度**：默认500ms，后续扩展至最长1000ms以探索最佳窗口
- **电极选择**：排除感觉运动、听觉及无关额叶区，保留视觉相关皮层（见Table 7）

### 📊 评估指标
- **主指标**：Balanced Accuracy (**BA**)，解决类别不平衡问题
- **辅指标**：Cohen’s Kappa (**K**)，衡量相对于随机猜测的表现
- **显著性检验**：
  - 单被试层面：paired t-test（n=60，按session和seed配对）
  - 群体层面：Wilcoxon Signed-Rank Test + Bonferroni校正
- **交叉验证**：Leave-One-Recording-Out (LORO)，确保训练/测试无时间泄漏

### 🆚 基线方法对比
| 方法 | 描述 |
|------|------|
| **Baseline** | 标准监督训练（cross-entropy loss） |
| **Prototypical Network (ProtoNet)** | 小样本学习，episode-based训练 |
| **Mixup / Manifold Mixup** | 输入或潜空间插值增强 |
| **SimCLR** | 自监督对比预训练 + 冻结编码器微调 |

此外还比较了六种主流神经网络作为编码器：
- ResNet
- LSNet
- TCN
- EEGNet
- BiLSTM
- **Transformer**

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）最优配置性能（Transformer + Mixup + High-gamma）
| 指标 | 数值 |
|------|------|
| **Balanced Accuracy (BA)** | **0.683**（单被试E02） |
| **Cohen's Kappa (K)** | **0.505** |
| **群体平均BA（n=17）** | **0.594**（900ms窗口） |
| **EVC子组最高BA** | **0.703**（n=9，V2-V4≥5电极） |

> 注：机会水平为 0.333，所有结果均显著高于 chance（p < 0.001）

#### （2）不同学习方法比较（E02）
| 方法 | BA | Kappa |
|------|-----|--------|
| Baseline | 0.435 | 0.151 |
| ProtoNet | 0.517 | 0.271 |
| Mixup | 0.522 | 0.269 |
| **SimCLR** | **0.565** | **0.335** |

虽然SimCLR性能最好，但因需额外预训练阶段且计算开销大，最终选用**Mixup**作为实用方案。

#### （3）不同编码器性能对比（High-gamma输入）
| Encoder | BA (E02) |
|--------|---------|
| ResNet | 0.544 |
| BiLSTM | 0.637 |
| **Transformer** | **0.683** ✅ |

Transformer在high-gamma频带上表现最佳。

#### （4）频带影响（Transformer, n=17）
| 频带 | 中位BA |
|-------|--------|
| Alpha | 0.358 |
| Beta | 0.379 |
| Low-gamma | 0.479 |
| Broadband-gamma | 0.503 |
| **High-gamma** | **0.527** ✅ |
| Broadband | 0.376 |

➡️ **high-gamma (80–150 Hz)** 是最具判别力的频带。

#### （5）时间窗大小的影响
| 窗口长度 | 平均BA（n=17） |
|---------|----------------|
| 500 ms | 0.535 |
| 700 ms | 0.567 |
| **900 ms** | **0.594** ✅ |
| 1000 ms | 0.584 |

➡️ **900ms** 是群体最优post-stimulus窗口。

### 🔬 消融实验结果
| 实验 | 发现 |
|------|------|
| **信号扰动实验** | Constant-RMS扰动导致BA下降 ~0.21（p<0.001），而相位打乱影响较小 ➜ 表明模型主要依赖**high-gamma谱功率**而非相位信息 |
| **注意力可视化** | Transformer在broadband输入下难以解析特定频带模式；但在分离频带输入时能捕捉周期性注意结构（如alpha ~100ms） |
| **多频带融合模型** | 将各频带分别编码后加权融合，使broadband输入的BA从0.403提升至0.539（p<0.001） |
| **区域重要性分析（permutation importance）** | 替换关键区域信号会导致性能显著下降，重要性排序如下：<br>1. Early Visual Cortex (V2-V4)<br>2. Lateral Temporal Cortex<br>3. Ventral Stream Visual Cortex<br>4. MT+ Complex & Neighboring Areas |

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **端到端DL框架可行**：即使在极少量标签数据（<50样本/类）下，也能有效从ECoG中解码动态视频的视觉语义。
2. **Transformer + Mixup + High-gamma 是最佳组合**：该配置不仅性能领先，且具备良好的可解释性。
3. **high-gamma频带最关键**：其谱功率携带最多任务相关信息，尤其在V2-V4等早期视觉皮层。
4. **足够长的时间窗至关重要**：**900ms** 的post-stimulus窗口优于传统的500ms，表明后期神经响应也参与语义加工。
5. **模型行为符合神经科学先验**：
   - 重视早期视觉皮层（V2-V4）、腹侧流（ventral stream）、MT+区和外侧颞叶皮层
   - 注意力机制反映真实神经振荡节律（如alpha ~100ms）
   - 利用的是神经活动的能量（power）而非精确相位

### ⚠️ 局限性
1. **数据来源限制**：来自癫痫患者，可能不适用于健康人群
2. **电极覆盖异质性强**：部分被试缺少关键区域（如V1/V2）覆盖，影响结果一致性
3. **固定时间窗设计**：未考虑个体或刺激特异性的响应延迟
4. **标签可能混杂低级视觉特征**：如Text类天然具有高对比度边缘，可能导致模型依赖低级线索
5. **解释性方法基于模型本身**：注意力权重、扰动分析等是模型内部机制的反映，不能直接等同于真实神经机制

### 🔮 未来工作方向
1. **开发更灵活的时间建模机制**：如动态窗口选择、事件触发分割
2. **探索跨被试通用解码器**：减少个性化训练需求
3. **结合vision-language模型**：实现更细粒度的语义解码（如句子级别）
4. **实时在线解码系统构建**：推动临床BCI应用落地
5. **改进融合策略**：当前multi-band fusion未超越单一high-gamma，需更先进的跨频带整合机制

---

> 💡 **总体评价**：  
> 本文展示了**高效、可靠、可解释的端到端ECoG语义解码框架**，在方法严谨性和神经科学合理性之间取得了良好平衡，为未来面向临床应用的BCI系统提供了重要参考路径。

</details>

---

### 11. [Conservative Query and Adaptive Regularization for Offline RL Under Uncertainty Estimation](https://arxiv.org/abs/2607.19199)

**Authors**: Li-Rong Zhou, Qin-Wen Luo, Sheng-Jun Huang  
**Category**: cs.LG  
**Published**: 2026-07-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.19199v1  

#### Abstract
Offline reinforcement learning (RL) aims to learn an effective policy from a static dataset, but its performance is fundamentally limited by dataset coverage. Action preference queries leverage expert feedback without additional environment interaction, enabling policy improvement during offline tra...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Conservative Query and Adaptive Regularization for Offline RL Under Uncertainty Estimation**

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

在 **Offline RL** 中，策略学习受限于离线数据集的覆盖范围，导致难以探索潜在更优的动作。虽然引入 **action preference query** 机制可以通过专家反馈提升策略性能，但现有方法存在两个关键缺陷：

- **Query Shift 问题**：查询的动作可能远离数据分布（OOD），导致价值函数估计偏差，引发不稳定的 Bellman 更新。
- **静态正则化限制**：偏好信息被简单地作为硬约束使用，缺乏对不确定性变化的动态适应能力，抑制了后期学习潜力。

此外，现有方法（如 OAP）难以与 **value regularization** 类算法（如 CQL）有效结合，限制了其适用性。

---

### **提出了什么新方法或新思路**

本文提出了一种轻量级、通用性强的新框架：**Conservative Query and Adaptive Regularization under Uncertainty Estimation**，核心包括两大机制：

#### ✅ **(1) Uncertainty-Driven Conservative Query（基于不确定性的保守查询）**

- 利用 **Morse Neural Network** 对状态-动作对进行 **uncertainty estimation**，量化其是否属于 in-distribution。
- 在生成候选查询动作时，仅保留 Morse Score 高于阈值（由数据集均值和标准差自适应设定）的动作，确保所查询动作靠近数据流形。
- 这种“保守”策略防止了 OOD 动作引起的 **value overestimation** 和训练不稳定。

#### ✅ **(2) Uncertainty-Aware Adaptive Regularization（不确定性感知的自适应正则化）**

- 不再使用固定的正则化强度，而是根据当前策略动作的 **OOD 程度动态调整**。
- 定义自适应正则系数：
  $$
  \alpha(s,a) = (1 - M_b(s,a))^\beta \cdot e^{\mu / (\sigma)} \cdot \alpha_0
  $$
  其中 $M_b(s,a)$ 是 Morse Score，$\beta$ 控制非线性斜率。
- 当动作接近数据分布时，正则化减弱，允许更多依赖 **Bellman backup**；当动作高度 OOD 时，加强正则化以保证安全。

该方法实现了 **pessimistic constraint** 与 **optimistic Bellman update** 的动态平衡。

---

### **相比现有方法的优势**

| 维度 | 现有方法（如 OAP） | 本文方法（CQ²L） |
|------|---------------------|------------------|
| 查询策略 | 基于欧氏距离选择，易选中 OOD 动作 | 基于 uncertainty filtering，避免 query shift |
| 正则化方式 | 固定强度，静态约束 | 自适应调节，随训练进程演化 |
| 可扩展性 | 主要用于 policy constraint 方法（如 TD3+BC） | 成功集成到 value regularization 方法（如 CQL） |
| 性能增益 | 提升有限，尤其在复杂任务上 | 显著优于 baseline 与 OAP |

> 🌟 **核心优势**：首次将 action preference query 有效融入 **value regularization 框架**，提升了方法的通用性和稳定性。

---

## 2. 核心实验方法和设置

### **使用的数据集**

在 **D4RL benchmark** 上进行全面验证，涵盖两类典型任务：

- **MuJoCo Locomotion Tasks**  
  包括 `HalfCheetah`, `Hopper`, `Walker2d`，每类含三种数据集类型：
  - medium
  - medium-replay
  - medium-expert

- **AntMaze Navigation Tasks**  
  更具挑战性的长视野、稀疏奖励环境：
  - umaze / medium / large
  - play / diverse 变体

共 **9 个 MuJoCo 任务 + 6 个 AntMaze 任务**

---

### **实验设置和评估指标**

- **训练步数**：
  - Offline 阶段：1,000,000 步
  - Online fine-tuning（若启用）：250,000 步
- **偏好查询预算**：
  - CQ²L：最多 **90,000 次 preference query**
  - OAP：100,000 次（原文报告）
- **评估方式**：
  - 报告最终 10 次 eval 的平均归一化得分（normalized score）
  - 所有结果取 **5 个随机种子** 的均值 ± 标准差
- **实现细节**：
  - 使用 **CQL** 作为基础算法，命名为 **Conservative Query Q-Learning (CQ²L)**
  - Oracle 使用更强策略的 Q 函数判断偏好（通过 offline-to-online 方法训练）

---

### **基线方法对比**

分为两个方案进行比较：

#### 🔹 **Offline Scheme（纯离线）**
- AWAC, IQL, SPOT, Cal-QL, TD3+BC, CQL, CPI

#### 🔹 **Offline-to-Online (O2O) Scheme**
- 各方法 + 25万步在线微调
- OAP（唯一已有 preference query 方法）
- CQ²L（仅 9万次 query，无环境交互）

> ⚠️ 注意：OAP 未开源，采用原论文报告结果；其余方法使用官方或主流开源实现。

---

## 3. 主要实验结果和性能指标

### **关键性能数据（来自 Table 1 & 2）**

#### ✅ **MuJoCo 平均表现（归一化得分）**

| 方法 | Offline 平均 | O2O 平均 |
|------|-------------|---------|
| CQL | 79.1 | 82.39 |
| OAP | —— | 82.3 |
| **CQ²L** | **91.4** | **91.4** ✅ |

> 💡 CQ²L 在 **仅使用更少外部信息（90k vs 250k env steps）的情况下，达到甚至超越所有 fine-tuned baseline**。

#### ✅ **AntMaze 平均表现**

| 方法 | Offline 平均 | O2O 平均 |
|------|-------------|---------|
| CQL | 49.0 | 67.56 |
| OAP | —— | 48.6 |
| **CQ²L** | **75.8** | **75.8** ✅ |

> 🔥 在极具挑战的 AntMaze 上，CQ²L 表现尤为突出，显著领先于 OAP 和其他方法。

#### ✅ **代表性单项任务表现示例**

| Task | Best Baseline (O2O) | CQ²L |
|------|--------------------|-------|
| `hopper-medium-expert-v2` | 110.8 (CQL-ft) | **111.3** ✅ |
| `walker2d-medium-expert-v2` | 109.8 (CQL-ft) | **114.6** ✅ |
| `antmaze-large-diverse-v2` | 38.2 (CQL-ft) | **73.2** ✅ |

> 📈 多项任务中 CQ²L 超出 fine-tuned CQL 达数十个百分点，尤其在稀疏奖励任务中优势明显。

---

### **消融实验结果（Ablation Study）**

在四个代表任务上测试两个核心组件的影响：

| 组件 | 描述 | 影响 |
|------|------|------|
| **CQ（保守查询）** | 移除 Morse-based filtering | 在 AntMaze 上性能大幅下降（如 -15 pts），因引入 OOD 动作导致价值崩溃 |
| **AR（自适应正则）** | 改为固定正则系数 | 在 MuJoCo 上性能下降明显，说明无法充分利用 Bellman backup 的潜力 |

> 🔍 发现：
> - 在 **dense-reward、低维观测**（MuJoCo）任务中，**AR 更重要** → 更好利用 Bellman 更新。
> - 在 **sparse-reward、高维观测**（AntMaze）任务中，**CQ 更关键** → 防止无效探索和错误引导。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **Uncertainty estimation 是连接 preference query 与 value regularization 的桥梁**  
   Morse Network 提供可靠的 OOD 判断，使得 preference query 可安全用于 CQL 等算法。

2. ✅ **保守查询 + 自适应正则 = 稳定且高效的学习机制**  
   - 保守查询保障更新稳定性；
   - 自适应正则释放 Bellman backup 的潜力，实现“可信区域乐观更新”。

3. ✅ **少量高质量偏好即可超越大量环境交互**  
   CQ²L 仅用 **90,000 次 preference query** 就超过了多数经过 **250,000 步 online fine-tuning** 的方法。

4. ✅ **成功拓展 preference-based 方法至 value regularization 范畴**  
   首次证明 action preference query 可有效增强 CQL，打破原有方法局限于 policy constraint 的瓶颈。

---

### **方法的局限性**

- ❗ **依赖高质量 oracle**：偏好需来自可靠专家或强策略，否则会误导学习。
- ❗ **Morse Network 需预训练**：虽不增加训练负担，但在分布剧烈变化的任务中泛化能力待验证。
- ❗ **未考虑主动学习中的 query 成本优化**：当前 query 策略仍较被动，未来可引入信息增益等标准。

---

### **未来工作方向**

1. **将 CQ²L 扩展至其他 value regularization 方法**，如 Implicit Q-Learning (IQL) 或 Fisher-BRC。
2. **设计主动 query selection 策略**，基于 uncertainty 和 expected improvement 决定何时提问。
3. **结合 human-in-the-loop 设置**，研究真实人类偏好的噪声建模与鲁棒学习。
4. **探索 multi-step trajectory preference query**，进一步提升指导效率。

---

> ✅ **总体评价**：本文提出了一种简洁而强大的框架，在理论设计与实证效果之间取得了良好平衡，为 **offline RL + human feedback** 的融合提供了新范式。

</details>

---

### 12. [BatchDAG: LLM-Planned Execution Graphs for Scalable Ad-Hoc Analysis Over Enterprise Data](https://arxiv.org/abs/2607.18241)

**Authors**: Anupreet Walia  
**Category**: cs.AI  
**Published**: 2026-07-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.18241v1  

#### Abstract
Large language models (LLMs) excel at analyzing individual documents but break down on exhaustive, cross-entity analytical questions over enterprise-scale datasets due to context overflow, loss of per-entity attribution, and linear latency from sequential tool calls. We present BatchDAG, a system in...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：BatchDAG: LLM-Planned Execution Graphs for Scalable Ad-Hoc Analysis Over Enterprise Data**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**

当前基于 **tool-augmented LLM agents**（如 ReAct、LangChain）的方法在处理企业级大规模、跨实体（cross-entity）分析查询时面临三大瓶颈：

- **Context window overflow**：海量数据（如数万会议记录）超出 LLM 上下文窗口。
- **Loss of per-entity attribution**：全局检索无法将结果归因到具体实体（如特定交易或会议）。
- **Linear wall-clock latency**：顺序执行工具调用导致延迟随实体数量线性增长，难以扩展。

这类问题常见于企业场景中的“穷尽式分析”需求，例如：“检查每个销售机会是否讨论了安全保险”。

---

### 🚀 **提出了什么新方法或新思路**

提出 **BatchDAG** —— 一种两阶段架构系统：

1. **LLM Planner**：将自然语言查询转化为一个 **typed Directed Acyclic Graph (DAG)**，其中每个节点是预定义的操作类型（如 `sql`, `search`, `fan_out` 等）。
2. **Deterministic Execution Engine**：以拓扑波（topological-wave）方式并行执行 DAG，仅在必要节点调用 LLM。

#### 核心创新点：

| 创新点 | 描述 |
|--------|------|
| **Typed DAG Formalism** | 定义六种操作类型（`sql`, `search`, `transform`, `fan_out`, `analyze`, `compare`），其中只有 `fan_out` 和 `analyze` 需要 LLM 调用，其余为确定性操作，极大降低开销。 |
| **Entity-Aware Batching** | 在 `fan_out` 阶段按逻辑实体（如 meeting_id）分组批量处理，而非逐行分批，避免重复分析与上下文碎片化。 |
| **Structured JSON Data Flow** | 所有中间步骤传递结构化 JSON 数据，而非自然语言摘要，保障可追溯性（provenance）和组合性。 |
| **Topological-Wave Parallelism** | 并行执行无依赖的步骤波次（wave），显著减少端到端延迟。 |
| **Goal-Based Planning Prompt** | 引导 LLM 基于目标推理操作组合，而非模仿示例或硬编码规则，提升泛化能力。 |

---

### 🔍 **相比现有方法的优势**

| 维度 | BatchDAG | 现有方法（如 ReAct / 固定 Pipeline） |
|------|---------|-------------------------------|
| 可扩展性 | 支持 50K+ meetings 分析 | 通常限于单实体或小规模 |
| 成本效率 | 最多减少 **47× LLM 调用** | 每个实体都触发完整流程 |
| 归因能力 | 支持 **per-entity attribution** | 全局聚合，丢失细粒度归属 |
| 执行策略生成 | 自动从 NL 查询生成最优 DAG | 需手动设计多个专用 pipeline |
| 中间表示 | 结构化 JSON，防幻觉 | 多用 prose summary，易失真 |

> ✅ **核心思想转变**：  
> “**The LLM should plan, not execute.**”  
> 即：让 LLM 规划执行图，由系统高效执行，而非让 LLM 一步步“思考-行动”。

---

## 2. **核心实验方法和设置**

### 📊 **使用的数据集**

- **生产环境真实数据**（Brevian.ai 销售智能平台）：
  - **约 50,000 场会议**（meeting transcripts）
  - **3,000+ 销售机会**（opportunities/deals）
  - **平均每场会议 48 条 transcript rows**
  - 包含 CRM 数据、利益相关者信息、知识库文档等异构源

- **开发测试集**（用于控制实验）：
  - 46 场会议、62 个开放 deal、32 个账户
  - 用于验证黄金答案的精确查询

---

### ⚙️ **实验设置与评估指标**

| 类别 | 内容 |
|------|------|
| **查询类型** | 设计了 12 个高 transcript 密集型查询（TX01–TX12），需结合 SQL 元数据 + 文本证据进行判断 |
| **评估维度** | - 整体质量评分（1–5）<br>- **Transcript evidence rate**（引用原始文本支持的比例）<br>- Latency（响应时间）<br>- LLM 调用次数 / Token 开销<br>- 正确率（vs. 黄金标准） |
| **评分机制** | 使用 GPT-5.1 Judge（temperature=0.0, JSON mode）对答案打分，确保一致性 |
| **统计检验** | Paired t-test, Bootstrap 95% CI |

---

### 🆚 **基线方法对比**

| 基线系统 | 描述 |
|--------|------|
| **S5b: Enhanced Fan-out (Expert Pipeline)** | 手工优化的三阶段流水线：SQL → per-entity search + GPT-5.1 analysis → synthesis，代表最强 baseline |
| **S3: Intelligence (ReAct Agent)** | Brevian 当前使用的 ReAct 架构 agent，具备 SQL/search/analysis 工具调用能力 |
| **S6: Map-Reduce**, **S7: Long-Context** | 其他替代方案作为辅助比较 |

---

## 3. **主要实验结果和性能指标**

### 📈 **关键性能数据汇总**

| 指标 | BatchDAG | S5b (Enh. FO) | S3 (ReAct) |
|------|--------|-------------|-----------|
| **平均质量得分**（12 queries ×3 runs） | **3.74 / 5** | 3.25 / 5 | 3.09 / 5 |
| **Transcript 证据率** | **77%** | 46% | 60% |
| **Latency** | ~94 秒 | ~207 秒 | ~102 秒 |
| **LLM 调用数/查询**（transcript-heavy） | 25 | 24 | — |
| **Token 数/查询** | 102.1K | 287.3K | — |
| **成本/查询**（GPT-5.1 pricing） | **$0.24** | $0.68 | — |
| **全量分析耗时**（3K deals） | **<60 秒** | ~100 分钟（理论估计） | — |

> ✅ **显著优势**：  
> - 质量优于 ReAct Agent（p < 0.01）
> - 成本仅为专家 pipeline 的 **1/3**
> - 延迟降低 **2.2×**

---

### 🔬 **消融实验结果**

#### （1）**Entity-Aware Batching vs. Row-Level Batching**

| 指标 | Row-Level | Entity-Aware | 提升 |
|------|----------|--------------|-------|
| 总批次（batch size=5） | 1,165 | 25 | ↓ **47× 更少 LLM 调用** |
| 每实体上下文 | 片段化 | 完整会议内容 | ✔️ 支持整体分析 |
| 预估成本 | ~$70 | ~$1.50 | ↓ 47× 成本 |

> 💡 单一改进带来最大性能飞跃。

---

#### （2）**Structured JSON vs. Prose Summaries（中间表示对比）**

| 指标 | Structured JSON | Prose Summary |
|------|------------------|---------------|
| 平均质量 | 2.42 | 2.08 |
| 幻觉数/查询 | **10.9** | 14.9（↑27%） |
| LLM 调用数/查询 | 25.4 | 39.5（↑56%） |
| Token 数/查询 | 69K | 80K（↑16%） |
| W/T/L（质量胜率） | 9 赢 / 3 平 / 0 输 | — |

> ✅ 结构化中间表示显著减少 **hallucination**（↓27%），提高审计性和效率。

---

#### （3）**Planner 正确性测试（300 次规划调用）**

| 指标 | 结果 |
|------|------|
| 成功生成计划数 | 258 / 300（排除 API 错误） |
| 有效 DAG 率 | **98.8%**（255/258） |
| 失败原因 | 主要在复杂 `fan_out` 查询中出现 schema 错误（如列不存在） |

> ✅ 表明 goal-based prompt 设计稳定可靠。

---

## 4. **关键结论和发现**

### ✅ **主要发现**

1. **BatchDAG 不是精度提升器，而是通用编排层**：
   - 目标不是超越某个手工 pipeline 的准确率，而是**自动替换多个定制化 workflow**。
   - 对任意新查询都能自动生成合适的执行策略。

2. **结构化数据流至关重要**：
   - 使用 JSON 行而非 prose summary 是防止幻觉和保持 provenance 的关键。
   - 支持真正的数据库风格 join/filter/group 操作。

3. **Entity-aware batching 是性价比最高的优化**：
   - 通过识别“分析单位”（unit of analysis）实现高达 **47× 成本下降**。

4. **Goal-based prompting > Few-shot 或 Rule-based**：
   - 更能适应新颖查询，避免过拟合已有模式。

5. **实际部署表现优异**：
   - 在 **50K+ meetings 上分析仅需 <60 秒**
   - 单查询成本低至 **$0.02（SQL-only）~ $0.24（fan-out）**

---

### ⚠️ **局限性**

| 局限 | 说明 |
|------|------|
| **无前置验证机制** | 若 planner 生成错误 DAG，会完整执行后才发现问题。建议加入 probe phase（先试运行一批）。 |
| **静态 DAG 执行** | 不支持动态扩展（如 analyze 后追加新步骤），限制多跳推理能力。 |
| **缺乏公开基准** | 尚无标准 benchmark 用于 cross-entity analytical queries，影响横向对比。 |
| **大存储层级未充分验证** | >100K 行的数据暂存表（temp table）尚未在生产中广泛使用。 |

---

### 🔮 **未来工作方向**

1. **引入 Probe Phase**：在 full fan-out 前先验证一个小 batch，提前拦截错误 DAG。
2. **支持 Dynamic DAG Extension**：允许 analyze 步骤返回指令添加后续操作，增强灵活性。
3. **构建 Cross-Entity Analytical Benchmark**：推动该领域标准化评测。
4. **集成缓存与重用机制**：对相似查询复用部分 DAG 子图，进一步降本增效。

---

## ✅ **总结一句话**

> **BatchDAG 实现了“一次编写，处处运行”的企业级分析愿景：用一个系统代替数十个手工 pipeline，在保证高质量与强可追溯性的前提下，将跨实体穷尽分析的成本降低近 50 倍。**

</details>

---

### 13. [S2T-RLHF: Hierarchical Credit Assignment for Stable Preference-Based RLHF](https://arxiv.org/abs/2607.18258)

**Authors**: Wei Chen, Guanghui Zhu, Yafei Li, Limin Wang, Yihua Huang  
**Category**: cs.AI  
**Published**: 2026-07-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.18258v1  

#### Abstract
Reinforcement learning from human feedback (RLHF) with preference-based reward models often exhibits unstable training dynamics. A key contributing factor is that standard RLHF relies on a single sequence-level scalar reward, which is propagated to token-level policy updates and leaves credit assign...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# S2T-RLHF: Hierarchical Credit Assignment for Stable Preference-Based RLHF 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
标准的 **Reinforcement Learning from Human Feedback (RLHF)** 存在训练不稳定的问题，其根本原因在于：
- 奖励模型（Reward Model）仅提供一个**序列级（sequence-level）的标量奖励信号**。
- 该全局奖励被反向传播到每个 token 的策略更新中，导致**信用分配（credit assignment）模糊且方差高**。
- 这种“低分辨率”监督与语言生成的长时程、细粒度特性不匹配，容易引发**长度偏差、奖励集中、甚至奖励崩溃（reward collapse）**。

现有工作尝试通过将奖励细化至 token 级别来解决此问题，但作者指出：**过度细粒度的奖励分解会放大奖励噪声，反而加剧训练不稳定性**。

---

### 提出了什么新方法或新思路
作者提出了一种**基于语义层级的信用分配原则**，强调**稳定性优先于分配精度**，并据此设计了 **S2T-RLHF**（Sentence-to-Token RLHF）框架。

#### 核心思想：
- **奖励粒度应与语言的语义结构对齐**，而非一味追求更细粒度。
- **句子（sentence）是自然的中间粒度单元**，它既能保持语义连贯性，又能避免 token 级别的噪声干扰。

#### S2T-RLHF 方法概述：
S2T-RLHF 是一个两阶段的分层奖励分解框架：
1. **Stage I: 句子级奖励分配（Sentence-Level Allocation）**
   - 将序列级奖励按语义重要性分配给各个句子。
   - 使用 **Nash Bargaining**（纳什议价）机制进行公平协商，确保重要句子不被忽略。
   - 基于语义扰动（semantic perturbation）估计每个句子对整体奖励的影响。

2. **Stage II: 词元级奖励细化（Token-Level Refinement）**
   - 在每个句子内部，进一步将句子级奖励细化到 token 级别。
   - 使用轻量级的 **Dirichlet Token Allocation Network (DTAN)** 预测 token 权重。
   - 引入**有界（bounded）细化机制**，防止极端权重放大噪声。

> ✅ **无需重新训练奖励模型，也无需 token 级标注**。

---

### 相比现有方法的优势
| 维度 | S2T-RLHF | 传统方法（如 ABC, SCAR） |
|------|----------|------------------------|
| **稳定性** | 显著提升，减少早期崩溃风险 | 细粒度分解易放大噪声，导致不稳定 |
| **语义一致性** | 以句子为单位，保持逻辑完整性 | token 级分配可能破坏语义结构 |
| **鲁棒性** | 对学习率、KL 约束等超参数更鲁棒 | 在激进设置下易失效 |
| **实现成本** | 不需额外标注或模型重训 | 部分方法需辅助模型或复杂计算 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **主训练与分析数据集**：`HH-RLHF`（Helpful and Harmless RLHF），包含人类对回复的偏好比较。
- **零样本泛化评估数据集**：
  - `AdvBench`：对抗性提示，测试越狱能力。
  - `RealToxicityPrompts`, `ToxiGen`：毒性生成检测。
  - `SafeRLHF`：安全偏好数据集。

---

### 实验设置
- **模型架构**：
  - **Policy Model**: `Gemma-2-9B`
  - **Reward Model (训练用)**: `RM-Gemma-2B`
  - **Evaluator (评估用)**: `RM-Gemma-7B` 和 `GPT-4o-mini`（作为 Judge）
- **优化算法**：基于 `PPO` 的 RLHF 流程。
- **硬件环境**：单张 A100-SXM4-80GB GPU。

---

### 评估指标
1. **训练动态指标**（衡量稳定性）：
   - `KL Divergence`：策略偏离参考模型的程度。
   - `Policy Entropy`：探索程度，过低可能表示模式坍塌。
   - `Reward Trajectory`：训练过程中奖励的变化趋势。
   - `Obj-△`：目标函数步间变化绝对值（越小越稳定）。
2. **性能对齐指标**（衡量效果）：
   - **成对偏好评估（Pairwise Preference）**：
     - Win/Tie/Lose 比例，使用 `GPT-4o-mini` 或 `RM-Gemma-7B` 作为裁判。
3. **消融实验**：
   - 分离 `Sentence-only`, `Token-only`, `Full S2T` 架构。
   - 替换 Stage I 和 Stage II 的分配机制（如 uniform vs. Nash）。

---

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **RLHF** | 标准序列级标量奖励，无分解。 |
| **ABC** [30] | 基于注意力权重进行 token 级奖励密集化。 |
| **SCAR** [32] | 使用 Shapley 值分解 token 贡献。 |
| **S2T-RLHF (Ours)** | 本文提出的句子→词元两阶段分解。 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Tables 1–2, 11）

| 方法 | vs. RLHF (Win%) | vs. ABC (Win%) | vs. SCAR (Win%) | 数据来源 |
|------|------------------|----------------|------------------|---------|
| **S2T-RLHF** | **53.0%** > 36.5% | **51.5%** > 38.0% | **48.1%** > 43.1% | GPT-4o-mini |
| **S2T-RLHF** | **51.1%** > 39.5% | **48.7%** > 43.8% | **45.8%** > 44.1% | RM-Gemma-7B |

> ✅ 在所有基线上均取得显著胜率优势（Win > Lose），表明其偏好对齐能力更强。

---

### 与基线方法的对比结果
- **训练稳定性方面**：
  - S2T-RLHF 的 `KL divergence` 增长更平滑，**避免剧烈波动和早期崩溃**。
  - `Obj-△`（目标函数振荡）比 RLHF 降低一个数量级（见 Table 9–10）。
  - 在高学习率（如 1e-5）下，ABC 和 SCAR 出现**负奖励持续状态**，而 S2T-RLHF 仍保持稳定上升。
- **最终性能方面**：
  - S2T-RLHF 达到更高的最终平均奖励（Figure 3）。
  - 政策熵保持合理范围，无模式坍塌迹象（Figure 4）。
- **跨数据集泛化**：
  - 在 `AdvBench`, `ToxiGen`, `SafeRLHF` 上均优于 RLHF，尤其在安全性任务上表现突出（Table 11）。

---

### 消融实验结果（Tables 3–4, 13–14）
| 变体 | 结果 |
|------|------|
| **Sentence-only** | 比 Token-only 更早稳定，KL 控制更好，但最终性能略低。 |
| **Token-only** | 后期优化潜力大，但**早期极易不稳定**，KL 漂移严重。 |
| **Full S2T-RLHF** | **综合最优**：兼具早期稳定性和后期高性能。 |
| **Stage I 替换（Uniform / Length-proportional）** | Nash Bargaining 在 KL 和 Obj-△ 上全面优于其他方式。 |
| **Stage II 替换（Uniform / Linear+Softmax）** | DTAN 实现最佳稳定性-性能权衡。 |

> 🔍 发现：**两阶段结构本身 + 特定机制（Nash + DTAN）共同贡献了性能增益**。

---

## 4. 关键结论和发现

### 主要发现
1. **细粒度 ≠ 更好**：  
   过度追求 token 级信用分配会放大奖励噪声，**损害训练稳定性**。
   
2. **语义对齐优于纯技术优化**：  
   将奖励分解与语言的**句法-语义层级结构对齐**（句子→词元），能有效提升学习效率和鲁棒性。

3. **S2T-RLHF 实现了稳定性与性能的平衡**：
   - Stage I（句子级）提供**语义稳健的粗粒度引导**；
   - Stage II（词元级）引入**有界的局部细化**，保留表达精度而不失控。

4. **理论支持**：  
   附录证明 S2T-RLHF 缓解了三大 RLHF 不稳定源：
   - ✅ 打破全局优势对齐（Breaks Advantage Alignment）
   - ✅ 降低长程方差放大（Reduces Variance Amplification）
   - ✅ 解耦梯度尺度与序列长度（Decouples Gradient Scale from Length）

---

### 方法的局限性
1. **依赖预训练奖励模型**：无法纠正原始 Reward Model 的系统性偏见或错误。
2. **适用于开放生成任务**：对于强依赖符号推理的任务（如数学证明、代码执行），句子分割可能不够精细。
3. **计算开销增加**：主要来自 Stage I 的语义扰动估计（+81.5% 时间开销 vs. RLHF），虽可接受但仍高于基础方法。
4. **未覆盖所有任务类型**：实验集中在对话与通用对齐任务，未验证在摘要、翻译等任务上的普适性。

---

### 未来工作方向
1. **高效影响估计**：设计更高效的 sentence influence 估计算法（如缓存、批量处理、 amortized learning）。
2. **扩展至其他层级**：探索段落级或主题级的更高层次分解。
3. **结合过程监督**：在需要精确推理的任务中融合过程反馈（process-based feedback）。
4. **多模态扩展**：将层级信用分配思想应用于视觉-语言或多模态 RLHF。
5. **自动化粒度选择**：研究如何动态决定最优分解粒度，而非固定为“句子”。

---

> 📌 **总结一句话**：  
> S2T-RLHF 提出“**稳定性优先于精度**”的奖励设计哲学，通过**句子到词元的两阶段分层信用分配**，在不增加标注成本的前提下，显著提升了 RLHF 的训练稳定性和最终性能，为偏好对齐提供了新的范式。

</details>

---

### 14. [SkillSight: Seeing Through Shared Descriptions for Accurate Skill Retrieval](https://arxiv.org/abs/2607.18785)

**Authors**: Jinying Xiao, Bin Ji, Shasha Li, Xiaodong Liu, Ma Jun, Jiacheng Jie, Chao Wang, Nyima Tashi, Jie Yu  
**Category**: cs.AI  
**Published**: 2026-07-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.18785v1  

#### Abstract
As large language model agents gain access to increasingly large skill libraries, retrieving the right skill becomes critical to reliable capability selection and execution. Existing retrievers often treat skill descriptions as ordinary documents, overlooking their highly regular structure: shared d...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《SkillSight: Seeing Through Shared Descriptions for Accurate Skill Retrieval》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
在大型语言模型（LLM）代理系统中，随着可用技能（skill）数量的增长，从庞大的技能库中准确检索出符合任务需求的技能变得至关重要。然而，现有的**dense retriever**在处理技能文档时存在显著偏差。

技能文档具有高度结构化的共性描述模式（如接口说明、执行条件等），这些共享的“背景描述”（shared descriptive background）虽然对理解技能格式有帮助，但对区分不同技能的实际功能作用甚微。这种共性导致：
- 查询（query）与技能文档之间的语义相似度被虚假抬高；
- 出现“能量差距”（energy gap）——技能文档在嵌入空间中的背景子空间能量远高于查询；
- 掩盖了真正与任务相关的细粒度信号（如操作、目标实体、约束）。

因此，**现有检索器容易将与查询仅在通用描述上匹配的无关技能排到前列，造成误检**。

---

### **提出了什么新方法或新思路**
作者提出 **SkillSight**，一种无需训练的检索校准框架，通过在**语义表示**和**词法匹配**两个层面显式地校准共享背景信息，提升技能检索准确性。

#### **核心思想：分离任务相关信号与共享背景噪声**
SkillSight 包含两个互补模块：

1. **Semantic Background Calibration (SBC)**  
   - 利用 IDF 值识别出在整个技能语料库中频繁出现的**通用 token**（generic tokens）；
   - 构建一个由这些 token 和语料均值向量张成的**背景子空间**（background subspace）；
   - 将查询和文档的嵌入投影到该子空间的正交补空间，从而去除背景对相似度得分的影响；
   - 得分公式为：`s_(q,d) = [(I - BB^T)q]^T[(I - BB^T)d]`

2. **Lexical Evidence Calibration (LEC)**  
   - 在词法层面进一步强化判别性证据；
   - 移除查询中的通用 token 后，对剩余 token 使用加权覆盖评分：
     $$
     wp(t) = \text{IDF}(t) \cdot (1 - p(t))^\beta,\quad p(t) = \frac{\text{df}(t)}{N}
     $$
   - 引入 `(1 - p(t))` 因子进一步抑制跨文档高频词的影响；
   - 最终词法得分为非通用查询词在候选文档中的加权覆盖率。

最终融合两个通道的归一化得分：
$$
S_{\text{SkillSight}}(q,d) = s_(q,d) + [\tilde{s}_{\text{lex}}(q,d)]_+
$$

> ✅ **完全无需额外训练或推理开销**，所有背景子空间和文档投影均可离线预计算。

---

### **相比现有方法的优势**

| 维度 | SkillSight 优势 |
|------|----------------|
| **有效性** | 显著优于原始 dense retriever、BM25、ColBERTv2、SPLADE、ABTT、BGE-M3 Hybrid 等多种基线；超越两阶段 reranker 方法的性能 |
| **效率** | 检索延迟极低（仅 1.17–2.57 ms），比 Dense + Reranker 快 **677–1,248 倍**；远快于 SkillRouter |
| **通用性** | 可即插即用地应用于任何 embedding 模型（Qwen、SkillRouter、SkillRet 等） |
| **无训练依赖** | 不需要微调、重排序模型或图结构构建，部署成本极低 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**

| 数据集 | 规模 | 特点 |
|-------|------|------|
| **SRA-Bench** | 5,400 测试实例，约 26k 技能 | 多任务基准，涵盖 TheoremQA、LogicBench、ToolQA 等六个子任务，支持端到端评估 |
| **SkillBench-Supp** | ~77k 技能 | 更大规模技能库，用于测试不同检索规模下的表现 |
| （附录）**SkillRet** | —— | 用于补充验证泛化能力 |

---

### **实验设置和评估指标**

#### **离线检索评估指标**
- **Hit@k**：前 k 个结果中是否包含正确技能
- **Recall@k**：黄金技能出现在前 k 个的比例
- **MRR@k**：第一个正确结果的平均倒数排名

> 主要关注 **Recall@10**，反映检索覆盖面。

#### **端到端代理执行评估**
- 使用三种主流 agent 模型进行任务执行：
  - `Llama-3.1-8B-Instruct`
  - `GPT-5.4-mini`
  - `Qwen3-4B-Instruct`
- 输入为检索返回的 top-k 技能文档
- 输出任务完成率（Overall Score）
- 对比策略：
  - **LLM Selection**：LLM 从检索结果中选择技能
  - **Progressive Disclosure**：按需加载技能内容
  - **Oracle Skill**：直接提供黄金技能（上限参考）

#### **实现细节**
- 中间候选池大小 $ K = 300 $
- 参数 $\beta = 1$，背景子空间维度 $r$ 自动确定（基于谱有效秩）
- 所有文档嵌入和背景子空间离线预计算

---

### **基线方法对比**

| 类型 | 方法 |
|------|------|
| **稀疏检索** | BM25, TF-IDF Cosine, SPLADE |
| **密集检索** | Dense (Qwen3-Embedding), ABTT, SIF |
| **交互式/细粒度** | ColBERTv2 |
| **混合检索** | BGE-M3 Hybrid, Dense-BM25 RRF |
| **两阶段重排** | Dense + Reranker (Qwen3-Reranker) |
| **专用技能检索器** | SkillRouter（完整 pipeline）、SkillRet（embedding only） |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **表1：离线检索性能（SRA-Bench & SkillBench-Supp）**

| Method | SRA-Bench R@10 | SkillBench-Supp R@10 | Latency (ms) |
|--------|------------------|------------------------|--------------|
| Dense | 66.02 | 56.56 | 0.13 / 0.57 |
| Dense + Reranker | 77.18 | 56.83 | 1460 / 1740 |
| **SkillSight** | **86.23** (+20.21) | **64.24** (+7.68) | **1.17 / 2.57** |

> 🔺 **Recall@10 提升高达 20.21 个百分点！**

#### **表2：端到端代理性能（SRA-Bench Overall Score）**

| Agent Model | LLM Selection | SkillSight | 提升 |
|------------|---------------|-------------|------|
| Llama-3.1-8B | 33.83 | **38.80** | +4.97 |
| GPT-5.4-mini | 67.83 | **69.67** | +1.84 |
| Qwen3-4B | 56.85 | **61.13** | +4.28 |

> ✅ SkillSight 在所有模型上均取得**最佳非 Oracle 性能**

---

### **与专用技能检索器对比**

| 方法 | SRA-Bench R@10 | Latency (ms) | 效率倍数 |
|------|----------------|--------------|---------|
| SkillRouter | 70.39 | 1633.97 | —— |
| **SkillSight** | **86.23** | **1.17** | **快 1,397×** |

> ⚡ SkillSight 不仅性能更高，且速度超 SkillRouter 近三个数量级！

---

### **消融实验结果（Ablation Study）**

#### **组件有效性分析（SRA-Bench）**

| 方法 | Recall@10 |
|------|-----------|
| Dense | 66.02 |
| Dense + LEC | 79.76 |
| BM25 | 63.51 |
| BM25 + SBC | 75.41 |
| w/o LEC (SBC only) | 78.23 |
| w/o SBC (LEC only) | 71.46 |
| **SkillSight (完整)** | **86.23** |

> 📌 发现：
- **SBC 是主要贡献者**：去除后性能大幅下降（→71.46），说明共享背景是 dense ranking 的主要偏见来源；
- **LEC 改善排序精细度**：尤其在语义相近候选之间；
- 二者**互补性强**，联合使用效果最优。

#### **通用 token 选择策略对比**

| 策略 | Recall@10 |
|------|-----------|
| All tokens | 85.54 |
| Random tokens | 85.20 |
| High-IDF (Reverse) | 79.48 |
| **Low-IDF (SkillSight)** | **86.23** |

> ✅ 证明使用 **low-IDF token** 来估计背景是最优选择。

---

## **4. 关键结论和发现**

### **主要发现**

1. **共享描述是技能检索中的系统性偏见源**  
   - 技能文档普遍存在结构性共性（接口、条件、流程）；
   - 这些共性在 dense embedding 中形成“背景子空间”，导致文档侧能量显著高于查询；
   - 背景对齐会虚增相似度，掩盖任务相关信号。

2. **SkillSight 有效解耦背景与能力信号**
   - SBC 成功剥离语义空间中的背景干扰；
   - LEC 恢复细粒度词法判别力；
   - 两者结合显著提升召回率与排序质量。

3. **无需训练即可实现 SOTA 性能**
   - SkillSight 是纯 post-processing 方法；
   - 可无缝集成至任意 embedding pipeline；
   - 性能超越需训练/重排的复杂系统。

4. **高效且可扩展**
   - 推理延迟极低（<3ms）；
   - 比神经重排器快上千倍；
   - 适用于实时 agent 决策场景。

---

### **方法的局限性**

- **依赖高质量 embedding**：若初始 dense retriever 表现很差，SkillSight 提升有限；
- **静态背景建模**：当前背景子空间基于固定语料构建，难以适应动态更新的技能库；
- **未考虑技能间依赖关系**：如 GoSkills 或 SkillRouter 所做的图结构建模，不在本框架范围内。

---

### **未来工作方向**

1. **动态背景更新机制**：支持增量式技能添加后的背景子空间在线调整；
2. **联合优化 embedding 与背景校准**：设计 end-to-end 可学习的背景感知 embedding 模型；
3. **扩展至多模态技能检索**：如包含图像、代码片段的复合技能；
4. **探索更复杂的词法-语义融合策略**：例如引入 prompt-based lexical weighting。

---

> 💡 **一句话总结**：  
> SkillSight 揭示了技能检索中“共享描述”带来的系统性偏差，并提出一种无需训练、高效精准的双通道校准框架，在多个 benchmark 上实现了高达 **+20.21 Recall@10** 的提升，同时比最强 reranker 快 **1,248 倍**，为实用化 LLM agent 技能管理提供了强有力的新工具。

</details>

---

### 15. [Transcription Policy as a Latent Variable: Activating Controllable Verbatim ASR with Word-Level Timing](https://arxiv.org/abs/2607.18934)

**Authors**: Laurin Wagner (nyra labs), Mario Zusag (nyra labs), Bernhard Thallinger (nyra labs)  
**Category**: cs.CL  
**Published**: 2026-07-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.18934v1  

#### Abstract
Modern ASR models trained on heterogeneously annotated data treat transcription style (verbatim vs. intended) as an uncontrolled latent variable, causing measurable decoding instability, evaluation confounding (up to 60% of reported WER attributable to style mismatch), and unreliable word-level timi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Transcription Policy as a Latent Variable: Activating Controllable Verbatim ASR with Word-Level Timing*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题

现代自动语音识别（ASR）系统在训练时通常使用混合标注的数据（verbatim 与 intended 混合），导致 **transcription policy**（转录策略）成为一个**未受控的隐变量**（latent variable）。这引发三大问题：

- **解码不稳定性**（Decoding Instability）：同一段音频可能因 beam search 而产生显著不同的输出。
- **评估混淆**（Evaluation Confounding）：标准 WER 混淆了“内容错误”与“风格差异”，高达 60% 的 WER 可能源于风格不匹配而非识别失败。
- **词级时间戳定义不清**（Ill-defined Word-Level Timing）：由于模型对是否输出 disfluency 不确定，cross-attention 对齐变得不可靠。

此外，高质量的 **verbatim 转录数据稀缺**，尤其在非英语语言中。

---

### 🚀 提出的新方法与创新

本文提出三套互补机制，激活 ASR 模型中已存在但未被控制的能力：

#### （1）**Mode Tags：显式控制转录策略**

- 引入离散的 decoder 前缀 token（如 `[verbatim_1]`, `[intended_1]`）作为 **transcription policy** 的显式接口。
- 支持在 **verbatim**（保留填充词、重复、截断等）与 **intended**（仅保留流畅语义）模式间切换。
- 采用 **coverage-aware tag partitioning**：根据数据集标注覆盖情况动态分配标签（如只标 disfluency 不标 sound 的用 `[verbatim_1..3]`；只标 sound 的用 `[sound_1,2]`），避免因标注不全导致的梯度冲突。

> 💡 创新点：将隐式的转录风格变为可控制的显式变量，实现双向可控。

#### （2）**Supervised Cross-Attention：提升词级时间戳精度**

- 在 mode tags 稳定输出序列后，对 cross-attention 头进行**有监督训练**，使其学习对齐到真实词边界。
- 使用 MFA 生成的 word-level timestamps 构造二值目标 `gt`，最小化平均注意力与目标之间的 cosine distance。
- 推理时引入 energy-based pause model 和 attention sharpening，无需修改 tokenizer 即可实现高精度对齐。

> 💡 创新点：首次在 disfluent speech 上超越 forced alignment（如 MFA、WhisperX），且纯基于 attention，无外部组件。

#### （3）**Verbatimize：从 intended 转录重建 verbatim 转录**

- 给定音频 + 任意 intended transcript，模型可插入声学对齐的 disfluency（如 `[uh]`, 重复, cut-off）生成高质量 verbatim 转录。
- Prompt 格式：`[verbatim] <sot> intended_text <eot>`，目标为对应的 verbatim transcript。
- 引入 **casing perturbation**（随机大写提示中的关键词）迫使模型关注 prompt 内容拼写。

> 💡 创新点：实现低成本、可扩展的 verbatim 数据增强，打破“依赖人工标注”的瓶颈。

---

### 🔍 相比现有方法的优势

| 特性 | Whisper / WhisperX | CrisperWhisper | Reverb | 本文方法 |
|------|----------------------|----------------|--------|----------|
| 可控 verbatim/intended | ❌ | ❌ | ✅（连续参数，英文） | ✅✅（离散标签，多语言） |
| 多语言支持 | ✅ | ❌（需多语言 verbatim 数据） | ❌（仅英文） | ✅（零样本跨语言迁移） |
| 词级时间戳（disfluent speech） | ❌（依赖 WhisperX + FA） | ✅ | ❌ | ✅✅（优于 FA） |
| 声学事件标记（sounds） | ❌ | ❌ | ❌ | ✅ |
| Verbatim 数据生成能力 | ❌ | ❌ | ❌ | ✅（via verbatimize） |

> ✅ 本文实现了 **可控性、精确性、可扩展性** 的统一。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 数据集 | 语言 | 类型 | 用途 |
|-------|------|------|------|
| ICSI, AMI, CORAAL, NSC | EN | Verbatim + timestamps | 英文 verbatim 训练 |
| LibriSpeech, Common Voice | EN | Clean | 英文 intended 训练 |
| VocalSound, Nonspeech7k | Multi | Sound events | 注入 sound 进 clean 数据 |
| DisfluencySpeech | EN | Verbatim + intended | 英文评估 |
| FluencyBank | EN | Disfluent + timestamps | 时间戳评估 |
| TIMIT | EN | Read + timestamps | 时间戳基准 |
| Thorsten | DE | Read + timestamps | 德语时间戳评估 |
| GDS (作者构建) | DE | Verbatim + intended | 零样本德语评估 |
| ICSI rare-word set | EN | 含专有名词 | verbatimize 评估 |

> 所有数据均构造了 **paired verbatim/intended transcript** 对。

---

### 🧪 实验设置

- **基础模型**：OpenAI Whisper-medium
- **新增 token**：
  - Filler: `[uh]`, `[um]`
  - Sounds: `[laughter]`, `[cough]`, `[sigh]` 等共 12 个
  - Mode tags: `[verbatim_1..3]`, `[sound_1..2]`, `[intended_1..5]`
  - Verbatimize delimiters: `<sot>`, `<eot>`
- **两阶段训练**：
  1. **Stage 1**：冻结 encoder/decoder，仅训练新 token embeddings（1 epoch）
  2. **Stage 2**：解冻 decoder，继续 fine-tune（3 epochs）
- **Timing loss 权重**：0.2 × CE loss

---

### 📊 评估指标

| 能力 | 指标 |
|------|------|
| 转录质量 | vWER（verbatim WER）、iWER（intended WER）及其分解（iSR/iDR/iIR） |
| Disfluency 检测 | eF1（event F1）、fF1（filler）、sF1（sound）、cF1（cut-off）、rF1（repetition） |
| 时间戳精度 | MAE（Mean Absolute Error, ms）、F1@X ms（50/100/200ms 容差） |
| Verbatimize | CLR（Content Loss Rate）、RWR（Rare-Word Recall） |
| 解码稳定性 | Beam Divergence（10 beams 的最大成对 CER 中位数） |

---

### 🆚 基线方法对比

- **Whisper-medium** [13]
- **Canary-1B** [31]（NVIDIA 多语言 ASR）
- **Reverb** [23]（continuous verbatimicity，英文）
- **AssemblyAI Universal-3-Pro** [53]
- **CrisperWhisper** [19]（Whisper verbatim fine-tune）
- **WhisperX** [22] + **MFA** [51]（forced alignment 基线）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）**Disfluency Detection（表5）**

| 模型 | 英文 eF1 | 德文 eF1（零样本） |
|------|---------|------------------|
| Whisper | 12.0% | 10.3% |
| CrisperWhisper | 73.2% | 60.0% |
| **S1（仅训 tag embedding）** | **53.0%** | **78.9%** ✅ |
| **FTo（全 fine-tune，无德文数据）** | **90.7%** | **93.8%** ✅ |
| **FT10k（含德文数据）** | 95.0% | 95.2% |

> ⚡ **仅训练 27 个新 token embedding（Stage 1）即可使德语 eF1 从 10.3% → 78.9%**，证明模型已有能力，只需“激活”。

#### （2）**Intended Transcript 质量（表6）**

| 模型 | 英文 iIR（插入率） | 德文 iIR |
|------|------------------|--------|
| Whisper | 10.1% | 4.2% |
| CrisperWhisper | 16.5% | 18.4% |
| **FTo + tags** | **4.2%** | **3.6%** |
| **FTo - tags** | 15.9% | 15.9% |

> ✅ Mode tags 显著抑制 disfluency 泄漏到 intended 输出中（iIR ↓ 70%+）。

#### （3）**Word-Level Timing（表7）**

| 方法 | TIMIT MAE (ms) | FluencyBank MAE (ms) | Thorsten MAE (ms) |
|------|----------------|----------------------|------------------|
| Base Whisper | 203 | 568 | 218 |
| MFA | 19 | 142 | — |
| WhisperX | 66 | 200 | 89 |
| CrisperWhisper+s | 47 | 122 | 57 |
| **Ours (ah)+s** | **36** | **102** ✅ | **55** ✅ |

> ✅ **在 disfluent speech 上首次超越 forced alignment 基线（102ms vs 142ms）**  
> ✅ **零样本德语 timing 达 55ms MAE，优于 WhisperX（89ms）**

#### （4）**Verbatimize（表8）**

| 模型 | CLR ↓ | RWR ↑ |
|------|------|------|
| B（baseline） | 9.4% | 6.8% |
| B+V（verbatimize） | 1.5% | 94.1% |
| **B+V+C（+casing perturb）** | **1.3%** | **96.1%** ✅ |

> ✅ Verbatimize 将 rare-word 回收率从 6.8% → 96.1%，可用于大规模语料 enrich。

#### （5）**Beam Divergence（表2）**

| 模型 | AMI（disfluent）CER |
|------|---------------------|
| Whisper | 15.1% |
| Canary-1B | 14.6% |
| **Ours (verbatim)** | **8.1%** ✅（↓46%） |

> ✅ 显式 policy 控制显著降低解码不确定性。

---

### 🔬 消融实验结果

- **Mode tags 是关键**：相同模型（FTo）加 tag vs 不加 tag，德语 eF1 差 **34 个百分点**（93.8% vs 60.1%）。
- **Supervised attention 优于 per-head loss**：使用 averaged-head loss 比 per-head loss 更优（TIMIT: 36ms vs 45ms）。
- **Inference-time sharpening 有效**：无需 retrain，T=3 温度缩放提升所有 attention-based 方法。
- **Casing perturbation 提升 verbatimize**：强制模型关注 prompt 拼写，RWR 提升近 2%。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Transcription policy 是当前 ASR 的核心隐变量问题**，导致解码不稳定、评估失真、时间戳不可靠。
2. **大型 ASR 模型已编码 verbatim 与 intended 两种能力**，但缺乏显式控制信号。
3. **仅通过训练少量 mode tag embeddings（Stage 1）即可大幅激活能力**，证明是“激活”而非“从头学习”。
4. **显式 policy 控制 + supervised attention 可联合解决 timing 与 transcription 耦合问题**。
5. **Verbatimize 实现了从 clean transcript 到 verbatim 的高质量重建**，为语料库建设提供新范式。

---

### ⚠️ 局限性

- **跨语言泛化仅验证于德语**（与英语相近），远距离语言（如中文、阿拉伯语）尚待测试。
- **GDS 德语评估集由作者录制**，disfluency 自然性可能不如真实场景。
- **Timing 监督依赖 MFA 自动生成边界**，非人工标注，可能存在误差传播。

---

### 🔮 未来工作方向

- 将 mode tags 与 **speaker diarization embeddings** 结合，实现风格 + 角色联合控制。
- **从零训练专用 alignment heads**，而非复用 cross-attention。
- 引入 **VAD 或 pause detection 损失**，进一步提升 timing 精度。
- 使用 **beam search + heuristic filtering** 实现大规模 verbatim 语料 bootstrapping。

---

## 总结

> 本文揭示了 **transcription policy 作为 latent variable** 的危害，并提出 **mode tags + supervised attention + verbatimize** 三位一体方案，首次实现了：
>
> - ✅ **可控的 verbatim/intended ASR**
> - ✅ **高精度词级时间戳（尤其在 disfluent speech）**
> - ✅ **零样本跨语言迁移能力**
> - ✅ **可扩展的 verbatim 数据生成**
>
> 实验表明，**激活已有能力比从头训练更高效**，为未来 ASR 系统设计提供了新范式。

</details>

---

### 16. [Mapping Without Graphs: Learning Coherence Traffic for Task Placement](https://arxiv.org/abs/2607.18879)

**Authors**: Guochu Xiong, Tianrui Ma, Weichen Liu  
**Category**: cs.DC  
**Published**: 2026-07-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.18879v1  

#### Abstract
Cache coherence is essential for communication in many-core Network-on-Chip (NoC)-based systems. As application scale and complexity increase, efficiently managing communication becomes increasingly challenging, making task mapping a key optimization technique. However, existing task mapping approac...

---

### 17. [Scalable and Efficient Joint Spiking Embedding Predictive Architecture for Large-Scale Dynamic Graphs](https://arxiv.org/abs/2607.18412)

**Authors**: Huizhe Zhang, Yuchang Zhu, Huazhen Zhong, Liang Chen, Zibin Zheng  
**Category**: cs.LG  
**Published**: 2026-07-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.18412v1  

#### Abstract
Dynamic graph learning aims to capture evolving structural and semantic patterns in real-world systems, such as fraud detection and recommender systems. Due to the scarcity of labeled data in real-world dynamic graphs, recent studies have introduced generative or contrastive paradigms (e.g., masked ...

---

### 18. [Adopting Reinforcement Learning with Verifiable Rewards for Molecular Generation](https://arxiv.org/abs/2607.19044)

**Authors**: Mingxuan Ouyang, Hao Lan, Wanyu Lin  
**Category**: cs.LG  
**Published**: 2026-07-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.19044v1  

#### Abstract
Leveraging large language models (LLMs) for molecular generation has shown remarkable potential in chemical and drug design. Current methods primarily rely on supervised training or fine-tuning with limited datasets, which are insufficient to capture complex molecular design objectives. While some a...

---

### 19. [ROMS-IMLE: A Minimalist Approach to Competitive Single-Step Generative Modelling](https://arxiv.org/abs/2607.19332)

**Authors**: Chirag Vashist, Ke Li  
**Category**: cs.LG  
**Published**: 2026-07-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.19332v1  

#### Abstract
Generative models have undergone many generations of evolution, from VAEs/GANs to diffusion/flow matching. Along the way, the underlying techniques have become more complicated and various beliefs about what drives strong empirical performance have taken hold. Due to the success of diffusion models ...

---

### 20. [Athena-Brain Technical Report: An Efficient Robot Brain for General Intelligence and Embodied Interactio](https://arxiv.org/abs/2607.18985)

**Authors**: Jialian Li, Junhong Liu, Yuchen Cao, Weiran Guo, Jiaming Song, Xutao Wang, Yi Zhao, Jiangpin Liu, Jie Chen  
**Category**: cs.AI  
**Published**: 2026-07-22  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.18985v1  

#### Abstract
Large language models (LLMs) have demonstrated remarkable capabilities in language understanding, reasoning, and world knowledge. As embodied agents become increasingly capable, there is a growing demand for compact models that can serve as an on-device brain, preserving the broad general intelligen...

---

### 21. [A Reinforcement-Learning-Augmented Liquid-Fueled Reactor Network Model for Predicting Lean Blowout in Gas Turbine Combustors](https://arxiv.org/abs/2607.19281)

**Authors**: Philip John, Eloghosa Ikponmwoba, Pinaki Pal, Opeoluwa Owoyele  
**Category**: cs.LG  
**Published**: 2026-07-22  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.19281v1  

#### Abstract
This study introduces a reinforcement learning (RL) framework for generating optimal liquid-fueled reactors to improve lean blowout (LBO) predictions in gas turbine combustors. Existing approaches for determining cluster boundaries rely on manual heuristics or distance-based metrics in the input spa...

---

### 22. [Phionyx: A Deterministic AI Runtime Architecture with Structured State Management and Pre-Response Governance](https://arxiv.org/abs/2607.18246)

**Authors**: Ali Toygar Abak  
**Category**: cs.AI  
**Published**: 2026-07-22  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.18246v1  

#### Abstract
We present Phionyx, a deterministic AI runtime architecture derived from the broader Echoism interaction framework that introduces a governance-first approach to AI engineering: treating large language model (LLM) outputs as noisy sensor measurements rather than direct decisions. Unlike probabilisti...

---

### 23. [Probabilistic Concept-Aware Steering for Trustworthy LLM Inference](https://arxiv.org/abs/2607.18259)

**Authors**: Brian Becker, Rui Chu, Yingjie Lao  
**Category**: cs.AI  
**Published**: 2026-07-22  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.18259v1  

#### Abstract
Steering vectors (SVs), an inference-time intervention technique for large language models (LLMs), guide the generation process by adding a concept-specific direction vector to intermediate activations during inference. However, existing SV methods frequently yield representation-incoherent behavior...

---

### 24. [Comparative Study of Multi-Agent Actor-Critic Algorithms in Parameterized Action Reinforcement Learning](https://arxiv.org/abs/2607.19117)

**Authors**: Ubayd Ali Bapoo, Clement N Nyirenda  
**Category**: cs.AI  
**Published**: 2026-07-22  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.19117v1  

#### Abstract
Parameterized action reinforcement learning has shown strong performance in environments requiring both discrete action selection and continuous parameterization. Prior work established the effectiveness of single-agent actor-critic algorithms - Greedy Actor-Critic (GAC), Soft Actor-Critic (SAC), an...

---

### 25. [ResearchArena: Evaluating Sabotage and Monitoring in Automated AI R&D](https://arxiv.org/abs/2607.19321)

**Authors**: Lena Libon, Ben Rank, Jehyeok Yeon, David Schmotz, Jeremy Qin, Daniel Donnelly, Derck Prinzhorn, Maksym Andriushchenko  
**Category**: cs.AI  
**Published**: 2026-07-22  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.19321v1  

#### Abstract
As AI agents begin to automate AI R&amp;D, we need ways to assess whether their outputs are safe to deploy, even when the agents themselves may be untrusted. AI control offers one such approach: rather than trusting the agent, it treats it as a potential adversary and uses a monitor to detect covert...

---

### 26. [CASE: Causal Alignment and Structural Enforcement for Improving Chain-of-Thought Faithfulness](https://arxiv.org/abs/2607.18820)

**Authors**: Ziming Wang, Yinghua Yao, Changwu Huang, Ke Tang, Xin Yao  
**Category**: cs.CL  
**Published**: 2026-07-22  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.18820v1  

#### Abstract
Chain-of-thought (CoT) reasoning is widely used to improve both the performance and interpretability of large language models (LLMs), yet the generated reasoning may not faithfully support the final answer. We study this problem from a causal perspective, where a faithful CoT process should follow t...

---

### 27. [Predictive RTO for CoAP using Lightweight Support Vector Regression in Internet of Things](https://arxiv.org/abs/2607.18273)

**Authors**: Tobias Hansson, Praveen Kumar Donta  
**Category**: cs.DC  
**Published**: 2026-07-22  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.18273v1  

#### Abstract
Internet of Things (IoT) networks require lightweight application layer messaging, and CoAP is an option because it supports REST-style interactions over UDP on constrained devices. However, CoAP congestion control still depends on fixed heuristics, including binary exponential backoff (BEB) and RTT...

---

### 28. [On the Limits of Support-Preserving Alignment and Bounded Filtering](https://arxiv.org/abs/2607.18295)

**Authors**: Aryan Dutt, Rui Mao, Anupam Chattopadhyay  
**Category**: cs.LG  
**Published**: 2026-07-22  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.18295v1  

#### Abstract
We study whether alignment schemes that reshape a base model's output distribution, combined with bounded safety filters, can drive the probability of harmful behavior to zero in modern large language models. Recent research suggests that harmful behaviors can persist under preference-based alignmen...

---

### 29. [AHEAD: Advancing Multi-Class Label Aggregation with Interpretable Cross-Annotator Modeling](https://arxiv.org/abs/2607.18465)

**Authors**: Ju Chen, Sijia Xu, Jun Feng, Zhiqiang Gao, Zhengyi Yang  
**Category**: cs.LG  
**Published**: 2026-07-22  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.18465v1  

#### Abstract
Crowdsourced labeling provides valuable labeled data for domains across natural language processing, computer vision, and video. Label aggregation aims to infer latent true labels from noisy and biased annotations, with the key lying in annotator reliability estimation. Despite promising progress, e...

---

### 30. [From Agent Failure Paths to Quantified Residual Risk: A Compositional Framework for Resilient Agentic AI](https://arxiv.org/abs/2607.18243)

**Authors**: Hassan Karim, Sai Sitharaman, Deepti Gupta, Danda B. Rawat  
**Category**: cs.AI  
**Published**: 2026-07-22  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.18243v1  

#### Abstract
Agentic AI is crossing trust boundaries faster than current risk models can represent. Existing approaches provide one of two partial views. They either describe failure mechanisms without producing a transferable residual-risk estimate, or they produce a risk estimate while treating the internal fa...

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

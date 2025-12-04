# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2025-12-04 05:20:18 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Fantasy: Efficient Large-scale Vector Search on GPU Clusters with GPUDirect Async](https://arxiv.org/abs/2512.02278)

**Authors**: Yi Liu, Chen Qian  
**Category**: cs.DC  
**Published**: 2025-12-03  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2512.02278v1  

#### Abstract
Vector similarity search has become a critical component in AI-driven applications such as large language models (LLMs). To achieve high recall and low latency, GPUs are utilized to exploit massive parallelism for faster query processing. However, as the number of vectors continues to grow, the grap...

#### AI Summary (by kimi-k2-thinking)
# Fantasy: Efficient Large-scale Vector Search on GPU Clusters with GPUDirect Async

## 1. 论文的主要贡献和创新点

### **解决的问题**
- **单GPU内存容量限制**：大规模向量数据库（如十亿级向量）的图索引结构远超单GPU HBM容量（通常40-80GB），无法完全加载到GPU内存
- **CPU-GPU架构瓶颈**：现有CPU-GPU异构方案需将向量存储在CPU内存或SSD中，数据加载过程会阻塞GPU计算，导致GPU利用率低下
- **通信效率低下**：传统GPU间通信依赖CPU控制路径，产生额外的内存拷贝和同步开销

### **提出的新方法**
- **Fantasy系统架构**：在GPU集群上分布式存储向量数据库，每个GPU持有一个图分区和对应向量，完全驻留在HBM中
- **GPUDirect Async (IBGDA) 通信**：利用InfiniBand GPUDirect Async技术实现GPU直接发起RDMA操作，绕过CPU控制路径，NIC可直接访问GPU内存
- **两阶段微批处理流水线**：将向量搜索的四个阶段（K-means分类、向量分发、并行搜索、结果合并）编排成重叠执行的流水线，最大化GPU集群利用率

### **相比现有方法的优势**
- **计算-通信重叠**：GPU SMs在通信期间保持空闲，NIC独立处理数据传输，显著提升端到端吞吐量
- **消除数据加载停顿**：In-HBM搜索避免PCIe/NVMe带宽瓶颈，支持更大batch size
- **高可扩展性**：分布式设计支持千亿级向量搜索，吞吐量随GPU数量线性扩展
- **低延迟**：RDMA直接通信达到线速传输（400Gb/s），NVLink intra-node带宽达600GB/s

---

## 2. 核心实验方法和设置

**⚠️ 重要说明**：提供的论文文本中**未包含实际实验结果部分**，仅提供了理论性能模型和延迟估算。以下是论文中提及的评估框架：

### **理论分析参数设置**
- **硬件配置**：NVIDIA A100 GPU，HBM带宽1.55TB/s，峰值计算性能156 TFLOP/s (TF32)
- **网络配置**：NVLink带宽600GB/s，RDMA带宽25GB/s per GPU
- **集群规模**：R=16 ranks（2节点×8 GPU）
- **向量维度**：d=1536（典型LLM embedding维度）
- **数据类型**：FP32/FP16
- **Batch size**：bs=10,000 queries per rank
- **图参数**：出度dg=32，迭代次数I=6，beam width w=6

### **评估指标**
- **各阶段延迟**：K-means分类、All-to-All分发、并行搜索、结果合并
- **吞吐量**：Queries Per Second (QPS)
- **算术强度**：FLOP/byte比值
- **带宽利用率**：HBM、NVLink、RDMA

### **基线方法对比（理论分析）**
- **Out-of-core搜索**：CPU内存/SSD → GPU，受BIO带宽限制（PCIe 4×16≈32GB/s）
- **In-HBM集体搜索**：纯GPU内存计算，受HBM带宽限制（~1.5TB/s）

---

## 3. 主要实验结果和性能指标（理论估算）

### **阶段级性能分解**

| 阶段 | 延迟估算 | 关键参数 | 性能数据 |
|------|---------|---------|---------|
| **Stage 1: K-means分类** | ~1.35 ms | bs=10k, C=4096, d=1536 | 计算密集型，TF32 Tensor Core利用率~60% |
| **Stage 2: All-to-All分发** | ~3.67 ms | bs=10k, c=3, d=1536 | 数据量：91.9MB/rank，RDMA传输占主导 |
| **Stage 3: 并行搜索** | **~68.5 ms** | c×bs=30k vectors | **QPS≈4.37×10⁵ queries/s per rank** |
| **Stage 4: 结果合并** | ~11.01 ms | c=3 | 约为分发阶段的c倍 |

### **核心性能指标**
- **单rank搜索吞吐量**：**437,000 queries/second**（受HBM带宽限制）
- **端到端延迟**：约**84.5 ms**（不含K-means的流水线重叠部分）
- **算术强度**：**0.5-0.75 FLOP/byte**（FP32），确认内存密集型特性
- **带宽瓶颈**：HBM带宽（1.55TB/s）为首要限制因素，而非计算能力

### **架构对比结论**
- **Out-of-core**：性能受BIO带宽限制（32-64GB/s），batch size增大时性能下降
- **In-HBM集体搜索**：可支持**大10-20倍**的batch size才达到带宽饱和，吞吐量显著更高

---

## 4. 关键结论和发现

### **主要发现**
1. **内存密集型本质**：图搜索算法（CAGRA/HNSW）算术强度极低（<1.5 FLOP/byte），性能几乎完全取决于内存带宽而非计算能力
2. **通信计算重叠是关键**：GPUDirect Async使GPU SMs在RDMA传输期间可执行其他计算，消除传统同步等待开销
3. **K-means路由高效**：基于K-means的查询路由可在**亚毫秒级**完成，分区开销可忽略
4. **RDMA带宽是主要瓶颈**：跨节点通信（25GB/s）占分发延迟的**>90%**，NVLink intra-node传输可忽略

### **方法局限性**
- **理论模型假设理想条件**：未考虑网络拥塞、同步开销、负载不均衡等实际因素
- **分区质量依赖**：K-means聚类效果直接影响查询召回率，未讨论c值选择对recall的影响
- **硬件依赖性强**：依赖IBGDA和NVLink等高端硬件，在普通以太网环境不适用
- **缺乏真实验证**：**无实际实验数据**，所有结论基于理论估算和模拟

### **未来工作方向**
- **动态负载均衡**：根据查询分布动态调整分区分配
- **精度-性能权衡**：探索不同c值（top-c分区）对recall和吞吐量的影响
- **容错机制**：在GPU故障时保证搜索可用性
- **成本优化**：结合SSD分层存储处理超大规模数据集

---

## 总结

Fantasy提出了一种**理论上有前景**的GPU集群向量搜索架构，通过GPUDirect Async实现通信计算重叠，解决了单GPU内存容量限制。其核心创新在于**软件-硬件协同设计**，充分利用现代GPU和网络特性。然而，**论文缺乏实际实验验证**，所有性能数据均为理论估算，这是其最大不足。实际效果需等待后续完整版本或会议发表后的实验部分披露。

---

### 2. [AutoNeural: Co-Designing Vision-Language Models for NPU Inference](https://arxiv.org/abs/2512.02924)

**Authors**: Wei Chen, Liangmin Wu, Yunhai Hu, Zhiyuan Li, Zhiyuan Cheng, Yicheng Qian, Lingyue Zhu, Zhipeng Hu, Luoyi Liang, Qiang Tang, Zhen Liu, Han Yang  
**Category**: cs.CL  
**Published**: 2025-12-03  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2512.02924v1  

#### Abstract
While Neural Processing Units (NPUs) offer high theoretical efficiency for edge AI, state-of-the-art Vision--Language Models (VLMs) tailored for GPUs often falter on these substrates. We attribute this hardware-model mismatch to two primary factors: the quantization brittleness of Vision Transformer...

#### AI Summary (by kimi-k2-thinking)
# AutoNeural: Co-Designing Vision-Language Models for NPU Inference

## 1. 论文的主要贡献和创新点

### 解决的问题
- **硬件-模型不匹配**：现有SOTA Vision-Language Models (VLMs)为GPU设计，在NPU上部署时面临严重性能退化
- **量化脆弱性**：Vision Transformers (ViTs)的激活分布存在异常值，在INT4/8/16量化下精度急剧下降
- **内存I/O瓶颈**：Transformer的自回归注意力机制需要频繁的Key-Value (KV)缓存访问，导致NPU计算单元闲置，无法利用其高算术吞吐量

### 提出的新方法
**AutoNeural**：一个NPU原生的VLM架构，专为整数推理协同设计
- **视觉编码器**：采用MobileNetV5风格的骨干网络，使用深度可分离卷积，确保激活分布有界，实现稳定的INT4/8/16量化
- **语言模型骨干**：融合State-Space Model (SSM)原理与Transformer层，使用高效门控卷积实现线性时间复杂度，消除生成过程中的KV缓存开销
- **NPU感知训练框架**：集成量化感知微调(QAT)、混合精度约束和硬件对齐策略

### 相比现有方法的优势
- **量化鲁棒性**：视觉编码器量化误差降低**7倍**
- **延迟优化**：端到端延迟降低**14倍**，首token时间(TTFT)从1.4秒降至约100ms
- **吞吐量提升**：解码速度提升**3倍**（44 tok/s vs 15 tok/s）
- **上下文扩展**：支持**4倍**更长的上下文窗口（4096 vs 1024）
- **高分辨率支持**：可在NPU上实时处理**768×768**分辨率图像，而ViT基线因内存限制无法运行

---

## 2. 核心实验方法和设置

### 数据集
**训练数据**：
- **Infinity-MM**：44.8M高质量多模态指令数据，涵盖通用VQA、文档理解、图表推理、OCR等任务
- **自定义汽车数据集**：20万标注样本，包含4类座舱AI任务：
  - AI Sentinel (56K)：车辆安全监控（破坏行为检测）
  - AI Greeter (50K)：身份识别与车辆解锁
  - AI Car Finder (44K)：停车场定位
  - Safety Monitoring (50K)：乘客上下车安全提醒

**评估基准**：
- MMStar：核心视觉语言推理
- HallusionBench：幻觉检测
- MathVista_MINI：数学推理
- AI2D_TEST：图表理解
- OCRBench：文本识别

### 实验设置
- **硬件平台**：Qualcomm SA8295P NPU（汽车SoC）
- **量化配置**：
  - 视觉编码器：W8A16（8位权重，16位激活）
  - 语言模型：W4A16（4位权重，16位激活）
- **训练策略**：四阶段课程学习
  1. 阶段1：冻结编码器和LLM，仅训练投影层
  2. 阶段2：解冻所有参数，通用视觉任务训练
  3. 阶段3：指令特定微调
  4. 阶段4：领域特定QAT + 合成数据集成

### 基线方法对比
- **InternVL2-1B/2B**：ViT-Transformer架构
- **Qwen2-VL-2B/3.75B**：通用多模态模型
- **InternViT-Qwen**：InternViT + Qwen2.5-1.5B组合
- **MobileNet-Qwen**：MobileNetV5 + Qwen2.5-1.5B（消融实验）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（NPU实测）
| 指标 | InternVL 2B基线 | AutoNeural-VL | 提升倍数 |
| :--- | :--- | :--- | :--- |
| **视觉编码器延迟** (512×512) | ~1.4秒 | ~100ms | **14×** |
| **最大支持分辨率** | 448×448 | 768×768 | - |
| **首Token时间(TTFT)** | ~1.40s | ~100ms | **14×** |
| **解码吞吐量** | ~15 tok/s | ~44 tok/s | **2.9×** |
| **上下文长度** | 1024 | 4096 | **4×** |
| **量化误差(RMS)** | 3.98% | 0.562% | **7×** |
| **信噪比(SQNR)** | 28dB | 45dB | **+17dB** |
| **LLM困惑度** (FP16→W4A16) | - | 21.13→21.47 | +1.6% |

### 视觉编码器延迟对比
- **256×256**：28.0ms vs 163.3ms（5.8×加速）
- **512×512**：101.7ms vs 1415.0ms（14×加速）
- **768×768**：278.1ms vs **无法运行**（InternViT-300M超出NPU内存容量）

### 基准测试性能
| 模型 | 参数量 | MMStar | Hallusion | MathVista | AI2D | OCRBench | 平均分 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| InternVL2-1B | 0.94B | 64.09 | 54.26 | 40.30 | 45.47 | 75.70 | 55.96 |
| InternVL2-2B | 2.2B | 73.90 | 58.36 | 49.10 | 50.07 | 78.40 | 61.97 |
| Qwen2-VL-2B | 2.2B | 74.64 | 62.04 | 47.10 | 47.80 | 80.90 | 62.50 |
| **AutoNeural** | **1.47B** | **73.80** | **56.05** | **53.10** | **49.40** | **71.40** | **60.75** |

### 消融实验结果
- **InternViT-Qwen**：63.08平均分（InternViT + Qwen2.5-1.5B）
- **MobileNet-Qwen**：65.15平均分（MobileNetV5 + Qwen2.5-1.5B）→ **证明MobileNet编码器不劣于ViT**
- **InternViT-Liquid**：59.29平均分（InternViT + Liquid AI）→ **验证混合架构有效性**
- **AutoNeural**：60.75平均分（MobileNet + Liquid AI）→ **在NPU效率与精度间取得最佳平衡**

---

## 4. 关键结论和发现

### 主要发现
1. **NPU原生架构的必要性**：传统"GPU优先"的缩放策略在NPU上收益递减，量化脆弱性和内存带宽成为主要瓶颈
2. **MobileNet编码器的量化优势**：深度可分离卷积产生有界激活分布，在INT8/16下保持稳定的SQNR（45dB），RMS误差仅0.562%
3. **混合Transformer-SSM的有效性**：门控卷积层消除KV缓存，减少60%内存I/O，实现线性复杂度，同时保留Transformer层的上下文学习能力
4. **实际部署验证**：在Qualcomm SA8295P汽车NPU上实现座舱AI实时响应，支持高分辨率视觉理解

### 方法局限性
- **精度权衡**：在部分基准（如OCRBench）上略低于最大基线，牺牲少量精度换取显著效率提升
- **量化评估范围**：基准测试主要在FP16下进行，虽提供SQNR和困惑度分析，但缺乏全面的量化部署评估
- **平台特定性**：当前验证集中于Qualcomm NPU，其他NPU平台（如MediaTek、Apple）的通用性需进一步验证

### 未来工作方向
1. **车内真实场景验证**：在多样化驾驶条件下评估量化模型性能
2. **多平台扩展**：在更多NPU架构上进行端到端质量评估
3. **自动化架构搜索**：基于硬件配置文件的条件化架构搜索
4. **混合精度优化**：探索更优的量化方案以进一步减少精度损失
5. **生产环境部署**：在汽车座舱AI系统中进行大规模实际部署测试

---

**核心洞见**：在边缘设备上实现鲁棒的多模态智能，必须针对NPU约束重新思考模型拓扑结构。通过选择深度可分离卷积（视觉）和SSM（语言）等对齐NPU执行模型的算子，才能充分释放边缘加速器的潜力。

---

### 3. [From monoliths to modules: Decomposing transducers for efficient world modelling](https://arxiv.org/abs/2512.02193)

**Authors**: Alexander Boyd, Franz Nowak, David Hyland, Manuel Baltieri, Fernando E. Rosas  
**Category**: cs.AI  
**Published**: 2025-12-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2512.02193v1  

#### Abstract
World models have been recently proposed as sandbox environments in which AI agents can be trained and evaluated before deployment. Although realistic world models often have high computational demands, efficient modelling is usually possible by exploiting the fact that real-world scenarios tend to ...

#### AI Summary (by kimi-k2-thinking)
# 论文总结：From monoliths to modules: Decomposing transducers for efficient world modelling

## 1. 主要贡献和创新点

### 解决的问题
论文针对**世界模型（world models）**在AI安全和可解释性方面的核心挑战：现有高维世界模型通常是难以解析的"单体"（monolith），缺乏结构性透明度和计算效率。虽然现实世界的场景往往由模块化组件以稀疏交互方式构成，但缺乏系统性的理论框架来识别和利用这种模块化结构。

### 提出的新方法
论文建立了基于**transducer**的完整理论框架，实现了世界模型的**可逆组合与分解**：

- **统一表示框架**：将世界模型、智能体和环境统一表示为transducer（推广了POMDP的模型），作为输入-输出接口的规范描述
- **信息论分解诊断**：
  - **Intransducibility**：当潜变量可观测时，检测变量集合是否能由因果transducer生成
  - **Acausality**：当仅有观测数据时，量化接口违反非预期性（nonanticipatory）条件的程度
- **分解算法**：提出两种递归分解算法，将单体transducer分解为稀疏网络结构
- **粗粒化理论**：证明如何在保留可观测接口的前提下合并transducer网络中的模块，实现多尺度建模

### 相比现有方法的优势
1. **结构性透明**：从单体模型中提取功能模块，每个对应世界的不同子系统
2. **计算效率**：支持**分布式推理**，各子transducer可基于局部输入-输出历史独立更新信念
3. **可解释性**：分解结果直接映射到功能组件，便于分析和审计
4. **理论完备性**：在ε-transducer最小预测表示下，组合操作保持最小性不变

---

## 2. 核心实验方法和设置

**重要说明**：这是一篇**纯理论论文**，**不包含任何实验验证或实证研究**。论文完全聚焦于：
- 形式化定义和定理证明
- 信息论度量的理论性质分析
- 算法的形式化描述（伪代码）

**无实验组件**：
- ❌ 未使用任何数据集（模拟或真实）
- ❌ 无基线方法对比
- ❌ 无性能指标评估
- ❌ 无消融实验

论文的"验证"完全基于**数学证明**和**理论分析**，主要技术工具包括：
- 计算力学（computational mechanics）
- 信息论（互信息、条件互信息）
- 线性算子理论（Kronecker积表示）
- 概率图模型（DAG表示）

---

## 3. 主要理论结果（替代实验结果）

虽然缺乏实证数据，论文提供了关键**理论保证**：

### 核心定理
1. **定理1**：非预期性接口 ⇔ 存在transducer表示
   - 建立了transducer作为因果接口的完备表达能力

2. **定理2**：ε-transducer的组合保持最小性
   - 若T和U分别是接口I[Y|X]和I[Z|XY]的最小预测模型（ε-transducer），则其组合UT是复合接口I[YZ|X]的ε-transducer
   - **意义**：模块化构建不会损失预测最优性

### 算法复杂度
- **算法1（基于Intransducibility）**：需计算长历史上的互信息，实践中需要有限时域或平稳性假设
- **算法2（基于Acausality）**：仅需可观测数据，但同样面临高阶统计量估计挑战

### 理论性能指标
论文用**信息论量**作为"性能"度量：
- **Intransducibility = 0**：完美分解的判定条件
- **Acausality = 0**：因果序识别的判定条件
- **稀疏性**：分解后网络的边数减少程度

---

## 4. 关键结论和发现

### 主要发现
1. **可逆性**：transducer的组合操作可逆，在适当条件下可唯一分解为素（prime）组件
2. **因果序提取**：通过递归"剥离"最小输出集合，可恢复隐式的有向无环图（DAG）结构
3. **多尺度一致性**：粗粒化操作（条件化或边缘化）保持剩余变量的接口不变，支持跨尺度模型转换
4. **与神经网络的联系**：分解的因果状态对应于transformer等网络中的信念子空间，为机制可解释性提供理论基础

### 方法局限性
1. **计算可行性**：需要估计长历史的联合分布，高维情况下可能不可行
2. **理论假设**：
   - 仅限**前馈**（feedforward）系统，未处理反馈循环
   - 要求**机械平稳性**（mechanistically stationary），不适应非平稳环境
   - 分解不唯一，取决于打破平局（tie-breaking）的策略
3. **实证空白**：缺乏在真实世界模型（如RL环境、视频预测模型）上的验证

### 未来工作方向
1. **算法近似**：开发有限时域、变分近似和统计检验的实用版本
2. **扩展理论**：处理反馈、自适应和非平稳系统
3. **实证验证**：在训练好的世界模型、RL环境和大型神经网络上测试
4. **AI安全应用**：将框架用于智能体沙盒测试和内部表示审计

---

## 总结

该论文是**世界模型模块化**的奠基性理论工作，其价值在于提供了从单体到模块的形式化转换路径，而非实证性能提升。核心贡献是**信息论分解框架**和**保持最优性的组合理论**，为AI安全和可解释性研究提供了新的数学语言。实际应用需等待高效近似算法的开发和大规模实验验证。

---

### 4. [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556)

**Authors**: DeepSeek-AI, Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenhao Xu, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Erhang Li, Fangqi Zhou, Fangyun Lin, Fucong Dai, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Hao Li, Haofen Liang, Haoran Wei, Haowei Zhang, Haowen Luo, Haozhe Ji, Honghui Ding, Hongxuan Tang, Huanqi Cao, Huazuo Gao, Hui Qu, Hui Zeng, Jialiang Huang, Jiashi Li, Jiaxin Xu, Jiewen Hu, Jingchang Chen, Jingting Xiang, Jingyang Yuan, Jingyuan Cheng, Jinhua Zhu, Jun Ran, Junguang Jiang, Junjie Qiu, Junlong Li, Junxiao Song, Kai Dong, Kaige Gao, Kang Guan, Kexin Huang, Kexing Zhou, Kezhao Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Wang, Liang Zhao, Liangsheng Yin, Lihua Guo, Lingxiao Luo, Linwang Ma, Litong Wang, Liyue Zhang, M. S. Di, M. Y Xu, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingxu Zhou, Panpan Huang, Peixin Cong, Peiyi Wang, Qiancheng Wang, Qihao Zhu, Qingyang Li, Qinyu Chen, Qiushi Du, Ruiling Xu, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runqiu Yin, Runxin Xu, Ruomeng Shen, Ruoyu Zhang, S. H. Liu, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaofei Cai, Shaoyuan Chen, Shengding Hu, Shengyu Liu, Shiqiang Hu, Shirong Ma, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, Songyang Zhou, Tao Ni, Tao Yun, Tian Pei, Tian Ye, Tianyuan Yue, Wangding Zeng, Wen Liu, Wenfeng Liang, Wenjie Pang, Wenjing Luo, Wenjun Gao, Wentao Zhang, Xi Gao, Xiangwen Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaokang Zhang, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingkai Yu, Xingyou Li, Xinyu Yang, Xinyuan Li, Xu Chen, Xuecheng Su, Xuehai Pan, Xuheng Lin, Xuwei Fu, Y. Q. Wang, Yang Zhang, Yanhong Xu, Yanru Ma, Yao Li, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Qian, Yi Yu, Yichao Zhang, Yifan Ding, Yifan Shi, Yiliang Xiong, Ying He, Ying Zhou, Yinmin Zhong, Yishi Piao, Yisong Wang, Yixiao Chen, Yixuan Tan, Yixuan Wei, Yiyang Ma, Yiyuan Liu, Yonglun Yang, Yongqiang Guo, Yongtong Wu, Yu Wu, Yuan Cheng, Yuan Ou, Yuanfan Xu, Yuduan Wang, Yue Gong, Yuhan Wu, Yuheng Zou, Yukun Li, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Z. F. Wu, Z. Z. Ren, Zehua Zhao, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhibin Gou, Zhicheng Ma, Zhigang Yan, Zhihong Shao, Zhixian Huang, Zhiyu Wu, Zhuoshu Li, Zhuping Zhang, Zian Xu, Zihao Wang, Zihui Gu, Zijia Zhu, Zilin Li, Zipeng Zhang, Ziwei Xie, Ziyi Gao, Zizheng Pan, Zongqing Yao, Bei Feng, Hui Li, J. L. Cai, Jiaqi Ni, Lei Xu, Meng Li, Ning Tian, R. J. Chen, R. L. Jin, S. S. Li, Shuang Zhou, Tianyu Sun, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xinnan Song, Xinyi Zhou, Y. X. Zhu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, Dongjie Ji, Jian Liang, Jianzhong Guo, Jin Chen, Leyi Xia, Miaojun Wang, Mingming Li, Peng Zhang, Ruyi Chen, Shangmian Sun, Shaoqing Wu, Shengfeng Ye, T. Wang, W. L. Xiao, Wei An, Xianzu Wang, Xiaowen Sun, Xiaoxiang Wang, Ying Tang, Yukun Zha, Zekai Zhang, Zhe Ju, Zhen Zhang, Zihua Qu  
**Category**: cs.CL  
**Published**: 2025-12-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2512.02556v1  

#### Abstract
We introduce DeepSeek-V3.2, a model that harmonizes high computational efficiency with superior reasoning and agent performance. The key technical breakthroughs of DeepSeek-V3.2 are as follows: (1) DeepSeek Sparse Attention (DSA): We introduce DSA, an efficient attention mechanism that substantially...

#### AI Summary (by kimi-k2-thinking)
# DeepSeek-V3.2 论文核心总结

## 1. 主要贡献和创新点

### 解决的问题
- **开源模型与闭源模型性能差距扩大**：开源社区虽然持续进步，但闭源模型（GPT-5、Gemini-3.0-Pro）性能提升更快，在复杂任务中差距明显
- **三大核心缺陷**：
  1. **架构效率瓶颈**：传统vanilla attention机制在长序列场景下计算复杂度过高
  2. **后训练计算投入不足**：开源模型在后训练阶段计算资源分配不足，限制硬任务性能
  3. **Agent能力滞后**：开源模型在工具使用场景下的泛化和指令遵循能力显著落后于专有模型

### 提出的新方法
1. **DeepSeek Sparse Attention (DSA)**
   - 高效注意力机制，将核心注意力复杂度从 $O(L^2)$ 降至 $O(Lk)$（$k \ll L$）
   - 包含lightning indexer和细粒度token选择机制，仅选择top-k key-value对进行计算
   - 在MLA架构基础上实例化，采用MQA模式实现kernel级共享

2. **可扩展的强化学习框架**
   - 开发稳定的RL scaling协议，后训练计算预算超过预训练成本的10%
   - 采用Group Relative Policy Optimization (GRPO)算法
   - 引入多项稳定训练策略：无偏KL估计、Off-Policy序列掩码、Keep Routing、Keep Sampling Mask

3. **大规模Agent任务合成Pipeline**
   - 系统化生成训练数据：1,827个环境 + 85,000个复杂prompt
   - 冷启动阶段：统一推理和工具调用于单一轨迹
   - 多领域覆盖：代码、搜索、通用Agent、代码解释器等

### 相比现有方法的优势
- **计算效率**：DSA在长上下文场景实现显著端到端加速，128K序列推理成本降低
- **性能突破**：DeepSeek-V3.2-Speciale在IMO 2025和IOI 2025获得金牌，首次实现开源模型在顶级竞赛中达到金牌水平
- **Agent能力**：在工具使用基准上显著缩小开源与闭源模型差距，成本效益更高
- **泛化能力**：合成任务能有效迁移到真实环境，解决长尾Agent任务

---

## 2. 核心实验方法和设置

### 数据集与评估基准
**推理能力评估**：
- MMLU-Pro、GPQA Diamond、Human Last Exam (HLE)
- AIME 2025、HMMT 2025、IMO AnswerBench
- LiveCodeBench、Codeforces

**Agent能力评估**：
- Terminal Bench 2.0、SWE-Verified、SWE-Multilingual
- BrowseComp/BrowseCompZh、τ²-bench
- MCP-Universe、MCP-Mark、Tool-Decathlon

**长上下文评估**：
- AA-LCR、Fiction.liveBench

### 实验设置
- **模型变体**：DeepSeek-V3.2（标准版）、DeepSeek-V3.2-Speciale（高计算版）
- **上下文长度**：128K tokens
- **温度参数**：1.0
- **评估模式**：thinking mode（带推理链）和non-thinking mode（直接回答）
- **工具调用格式**：标准function calling格式，工具输出置于tool role消息中

### 基线方法
**闭源模型**：
- GPT-5-High、Claude-4.5-Sonnet、Gemini-3.0-Pro
- Kimi-K2-Thinking

**开源模型**：
- DeepSeek-V3.1-Terminus（架构对比基线）
- 其他支持thinking mode的工具使用模型

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 基准测试 | DeepSeek-V3.2 | GPT-5 High | Gemini-3.0 Pro | Kimi-K2 Thinking |
|---------|---------------|------------|----------------|------------------|
| **AIME 2025** | 93.1% (16k tokens) | 94.6% (13k) | **95.0%** (15k) | 94.5% (24k) |
| **HMMT Feb 2025** | 92.5% (19k) | 88.3% (16k) | 97.5% (16k) | 89.4% (31k) |
| **Codeforces** | 2386 Rating (42k) | 2537 (29k) | **2708** (22k) | - |
| **HLE** | 25.1% (21k) | 26.3% (15k) | **37.7%** (15k) | 23.9% (24k) |
| **LiveCodeBench** | 83.3% (16k) | 84.5% (13k) | **90.7%** (13k) | 82.6% (29k) |
| **SWE-Verified** | 73.1% | 74.9% | 76.2% | 71.3% |
| **Terminal Bench 2.0** | 46.4% | 35.2% | **54.2%** | 35.7% |
| **τ²-Bench** | 77.2% | 80.2% | **85.4%** | 74.3% |

### DeepSeek-V3.2-Speciale 突破性表现
- **IMO 2025**：35/42分（金牌线）
- **IOI 2025**：492/600分（金牌，排名第10）
- **ICPC WF 2025**：10/12题解决（金牌，排名第2）
- **CMO 2025**：102/126分（金牌）

### 架构效率对比
- **DSA vs Dense Attention**：在128K上下文下，prefilling阶段成本降低约50%，decoding阶段降低约30%
- **性能保持**：在短上下文和长上下文任务上均未观察到显著性能退化
- **AA-LCR基准**：DeepSeek-V3.2-Exp比V3.1-Terminus高4分

### 消融实验结果
1. **合成任务有效性**：
   - 随机50个合成任务：DeepSeek-V3.2-Exp仅12%准确率，前沿闭源模型最高62%
   - 仅使用合成数据RL：在τ²-bench、MCP-Mark等基准上显著提升，而仅code/search RL无改进

2. **Context Management策略**（BrowseComp基准）：
   - **Discard-all**：67.6分，效率与可扩展性最佳
   - **Summary**：60.2分，平均步骤扩展至364步
   - **无管理**：51.4分（20%+测试用例超出128K限制）

3. **Thinking vs Non-thinking模式**：
   - Terminal Bench：37.1% → 46.4%（+9.3%）
   - MCP-Mark：25.6% → 35.2%（+9.6%）
   - 非思考模式仍保持较强性能，适合特定框架（如Roo Code）

---

## 4. 关键结论和发现

### 主要发现
1. **计算投入scaling定律**：后训练计算预算超过预训练10%后，推理能力持续提升，表明RL scaling仍有巨大潜力
2. **稀疏注意力可行性**：DSA在保持性能的同时显著降低长序列计算成本，验证了稀疏模式的有效性
3. **合成数据价值**：大规模合成Agent任务能有效提升真实环境泛化能力，解决长尾任务
4. **推理与工具使用融合**：通过上下文管理策略（保留工具调用历史、丢弃推理内容）显著提升token效率

### 方法局限性
1. **知识广度不足**：总训练FLOPs少于前沿闭源模型，世界知识覆盖仍有差距
2. **Token效率待优化**：达到同等质量需要更长生成轨迹，智能密度低于Gemini-3.0-Pro
3. **复杂任务解决能力**：在部分高难度任务上仍落后于顶级闭源模型
4. **上下文管理兼容性**：当前thinking模式与Terminus等框架不兼容，需使用non-thinking模式

### 未来工作方向
1. **预训练规模扩展**：通过增加预训练计算量弥补知识广度差距
2. **推理链优化**：提升智能密度，减少token消耗，改善效率
3. **上下文管理优化**：探索串行与并行scaling的最佳组合策略
4. **基础模型改进**：持续优化基础模型能力和后训练方法
5. **Agent框架适配**：增强与现有Agent开发框架的兼容性

---

### 5. [Fast-Decoding Diffusion Language Models via Progress-Aware Confidence Schedules](https://arxiv.org/abs/2512.02892)

**Authors**: Amr Mohamed, Yang Zhang, Michalis Vazirgiannis, Guokan Shang  
**Category**: cs.CL  
**Published**: 2025-12-03  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2512.02892v1  

#### Abstract
Diffusion large language models (dLLMs) offer a promising alternative to autoregressive models, but their practical utility is severely hampered by slow, iterative sampling. We present SchED, a training-free, model-agnostic early-exit algorithm that aggregates full-span logit margins and halts decod...

---

### 6. [Unlocking the Power of Boltzmann Machines by Parallelizable Sampler and Efficient Temperature Estimation](https://arxiv.org/abs/2512.02323)

**Authors**: Kentaro Kubo, Hayato Goto  
**Category**: cs.LG  
**Published**: 2025-12-03  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2512.02323v1  

#### Abstract
Boltzmann machines (BMs) are powerful energy-based generative models, but their heavy training cost has largely confined practical use to Restricted BMs (RBMs) trained with an efficient learning method called contrastive divergence. More accurate learning typically requires Markov chain Monte Carlo ...

#### AI Summary (by kimi-k2-thinking)
# 论文总结：Unlocking the Power of Boltzmann Machines by Parallelizable Sampler and Efficient Temperature Estimation

## 1. 主要贡献和创新点

### 解决的问题
- **BM训练的计算瓶颈**：传统Boltzmann machines (BMs)依赖MCMC采样，其顺序更新规则导致训练缓慢且难以并行化
- **表达力与效率的权衡**：Restricted BMs (RBMs)可通过blocked Gibbs高效训练，但Semi-Restricted BMs (SRBMs)等更具表达力的模型因可见层单元间耦合而难以并行采样
- **逆温度估计难题**：基于优化机（如量子退火器、CIM）的并行采样器无法精确控制输出分布的逆温度βeff，导致学习过程不稳定

### 提出的新方法
1. **Langevin Simulated Bifurcation (LSB)**：一种受模拟分岔(SB)启发的并行化Boltzmann采样器
   - 基于Hamiltonian动力学方程，支持所有变量同时更新
   - 通过离散化和随机动量初始化增强采样多样性
   - 仅需两个超参数：时间步长Δ和噪声标准差σ

2. **Conditional Expectation Matching (CEM)**：高效的βeff估计方法
   - 利用SRBM中隐藏单元在可见层固定时的条件独立性
   - 通过最小化采样条件期望与解析表达式间的差异来估计βeff
   - 可与LSB采样过程并行执行，计算开销低

3. **Sampler-Adaptive Learning (SAL)**：整合LSB和CEM的统一学习框架
   - 动态适应采样器输出的有效逆温度βeff
   - 支持FBM、RBM和SRBM的高效训练

### 相比现有方法的优势
- **并行化能力**：LSB支持全变量并行更新，计算速度随问题规模线性扩展
- **采样精度**：在SRBM和SK模型上，LSB的采样精度DKL(PS||Bβeff)达到0.07±0.01，与Gibbs采样(0.09±0.02)相当甚至更优
- **表达力提升**：SRBMs在3-spin模型上显著优于RBMs，能捕捉高阶相关性
- **训练稳定性**：CEM的βeff估计误差仅3.6%，且偏差方向与MLPL相反，更适合SAL框架

---

## 2. 核心实验方法和设置

### 数据集
| 数据集 | 可见单元数Nv | 隐藏单元数Nh | 特点 | 评估方式 |
|--------|--------------|--------------|------|----------|
| **3-spin模型** | 10 | 5 | 具有三体相互作用的自旋玻璃系统，能量景观复杂 | 精确计算DKL(PD||Qβ) |
| **Bars-and-Stripes (BAS)** | 42 (7×6) | 21 | 192种条纹模式，测试生成与重建能力 | 视觉检查+像素错误率 |
| **OptDigits** | 74 (64像素+10类别) | 37 | 3823个训练样本，1797个测试样本，真实手写数字 | 分类准确率+视觉质量 |

### 实验设置
- **采样参数**：LSB迭代次数M=100（与CD-100公平对比），Δ=1，σ通过候选集{0.5,...,2.0}优化
- **训练配置**：学习率η=0.05(3-spin)或0.001(BAS/OptDigits)，动量α=0.5，L2正则化λ=10⁻⁵
- **评估指标**：
  - **KL散度**：DKL(PD||Qβ)衡量模型与数据分布差异
  - **采样精度**：DKL(PS||Bβeff)评估采样器输出与目标Boltzmann分布的匹配度
  - **重建错误**：缺失像素恢复的错误率
  - **分类准确率**：条件采样预测标签的正确率

### 基线方法对比
- **CD-k**：Contrastive Divergence with k steps (k=1,10,100)
- **Blocked Gibbs (BG)**：RBM的标准MCMC方法
- **MLPL**：Maximum Log-Pseudolikelihood估计βeff
- **DMFI**：CD结合阻尼平均场迭代用于SRBM

---

## 3. 主要实验结果和性能指标

### 3.1 3-spin模型（精确评估）
- **性能排序**：SRBM(SAL) > RBM(SAL) ≈ RBM(CD-100) > FBM(SAL)
- **关键数据**：
  - SRBM的DKL(PD||Qβ)在1000 epoch后降至~0.5，显著低于RBM的~1.0
  - SAL训练的RBM与CD-100性能相当，验证SAL有效性
  - FBM因无法建模三体相互作用，DKL高达~2.0

### 3.2 BAS数据集（生成与重建）
- **生成质量**：6000 epoch后LSB生成的36个样本**全部为有效BAS模式**，无错误生成
- **重建性能**：
  - 初始随机重建错误率：50.0%
  - 1000 epoch后错误率降至**0.5%**（几乎完美）
  - 3000 epoch后轻微上升至~2%，提示潜在过拟合

### 3.3 OptDigits数据集（大规模应用）
- **无条件生成**：3000 epoch后生成样本清晰可辨，数字结构准确
- **条件生成**：固定类别标签后，生成的数字与指定类别一致率达**90%以上**
- **分类性能**：
  - 初始随机准确率：10%
  - 500 epoch后达到**近90%**
  - 最终稳定在90%左右，与专用分类器相当

### 3.4 消融实验结果
- **CEM vs MLPL**：MLPL估计的βeff偏高，导致SRBM性能下降至RBM水平
- **CD在SRBM上**：CD-1导致DKL显著上升，CD-10/100+DMFI性能甚至低于RBM(CD-1)
- **采样器对比**：LSB在6/10个实例上优于Gibbs，平均DKL低22%

---

## 4. 关键结论和发现

### 主要发现
1. **SRBM的实用化**：SAL首次使SRBM训练变得可行且高效，在复杂分布建模上显著优于RBM
2. **LSB的优越性**：并行LSB采样精度与顺序MCMC相当，但计算效率更高，特别适合大规模问题
3. **CEM的关键作用**：准确的βeff估计是SAL成功的核心，CEM的并行性和精度使其成为理想选择
4. **通用性**：SAL框架适用于FBM、RBM和SRBM，其中SRBM展现了最强的表达力

### 方法局限性
- **理论基础不足**：LSB为何能良好逼近离散Boltzmann分布的理论解释尚不完整
- **超参数敏感**：σ需针对数据集手动调整，缺乏自动化选择策略
- **模型深度限制**：研究聚焦于单层SRBM，深层架构的扩展性待验证
- **隐藏层耦合**：含隐藏层内部连接的BM仍难以高效训练

### 未来工作方向
1. **理论深化**：研究LSB的连续极限与Langevin Monte Carlo的关系，建立采样精度的理论保证
2. **算法改进**：开发自适应超参数调整策略，探索LSB在更复杂能量函数（高阶项）上的应用
3. **架构扩展**：将SAL推广到Deep BMs和其他能量模型，研究在大规模数据集（如ImageNet）上的表现
4. **硬件实现**：利用GPU/TPU加速LSB的并行动力学演化，探索专用硬件实现可能性
5. **CEM分析**：深入研究CEM的统计性质、收敛条件和误差界

---

**核心贡献总结**：该工作通过LSB和CEM的协同设计，突破了BM训练的计算瓶颈，使SRBM这一更具表达力的模型首次在多个任务上展现出实用价值，为能量模型的发展开辟了新路径。

---

### 7. [On the Approximation of Phylogenetic Distance Functions by Artificial Neural Networks](https://arxiv.org/abs/2512.02223)

**Authors**: Benjamin K. Rosenzweig, Matthew W. Hahn  
**Category**: cs.LG  
**Published**: 2025-12-03  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2512.02223v1  

#### Abstract
Inferring the phylogenetic relationships among a sample of organisms is a fundamental problem in modern biology. While distance-based hierarchical clustering algorithms achieved early success on this task, these have been supplanted by Bayesian and maximum likelihood search procedures based on compl...

#### AI Summary (by kimi-k2-thinking)
# 论文总结：On the Approximation of Phylogenetic Distance Functions by Artificial Neural Networks

## 1. 主要贡献和创新点

### 解决的问题
- **系统发育推断的计算瓶颈**：传统基于距离的分层聚类算法计算效率高但准确性有限，而贝叶斯和最大似然方法虽然准确但计算开销巨大，难以扩展到大规模数据集（n>15个taxa，L>200bp）
- **深度学习在系统发育学中的应用困境**：现有深度学习方法要么需要极高容量的网络来记忆n!种排列对称性，要么缺乏对分子进化模型复杂性的理论指导

### 提出的新方法
- **最小化神经网络架构**：设计了专门用于近似经典系统发育距离函数的神经网络，包括：
  - **Sequence networks (S)**：将每个序列独立映射到嵌入空间，通过attention/convolution/pooling生成表示矩阵Z，输出欧氏距离或内积
  - **Pair networks (P)**：直接处理序列对，更具表达力但空间复杂度为O(n²)
- **对称性保持设计**：利用置换等变（permutation equivariant）层，确保网络对taxa排序不变，无需数据增强
- **理论指导架构设计**：基于Bourgain定理和Johnson-Lindenstrauss引理，论证了在低维欧氏空间中嵌入树度量的可行性

### 相比现有方法的优势
- **计算效率**：模型容量小（参数仅7K-400K），训练和推理内存占用低，可扩展至大规模数据集
- **泛化能力强**：在适当训练数据下，性能可与IQ-TREE等最先进方法媲美
- **模型复杂度匹配**：架构复杂度与进化模型复杂度相匹配，避免过度参数化

---

## 2. 核心实验方法和设置

### 数据集
- **模拟数据**：使用IQ-TREE的AliSim生成，基于birth-death模型（λ=1, μ=0.5, n=20）
- **进化模型**：
  - **JC** (Jukes-Cantor)：所有核苷酸频率相等，所有替换概率相同
  - **K2P** (Kimura 2-parameter)：转换和颠换不同权重
  - **HKY+GC**：带Gamma分布速率类别（连续Gamma(1,1)），可变转换/颠换比
  - **LG+GC**：Le-Gascuel蛋白替换矩阵，带Gamma速率类别
  - **LG+indel**：插入/删除率=0.01×替换率，indel长度服从几何分布

### 实验设置
- **序列长度**：训练集500 sites，测试集1000 sites（indel数据集最长5000 aa）
- **Taxa数量**：主要实验n=20，消融实验n=20-100
- **训练规模**：10⁵个样本，batch size=4

### 评估指标
- **Robinson-Foulds (RF) 距离**：未加权拓扑准确性（主要指标）
- **加权RF距离**：评估分支长度准确性（辅助分析）
- **度量性质验证**：检查输出是否满足三角不等式

### 基线方法
- **IQ-TREE**：提供正确模型规范的最大似然法
- **经典距离方法**：Hamming距离(dH)、Jukes-Cantor距离(dJC)、Kimura 2参数距离(dK2P)
- **现有深度学习方法**：Phyloformer（Transformer架构）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（表2）
**20-taxa树的平均RF距离（越低越好）**

| 方法 | JC | K2P | HKY | LG | LG+indel |
|------|-----|-----|-----|-----|----------|
| **dH** | 0.1244 | 0.1204 | 0.1032 | 0.0847 | 0.0739 |
| **dJC** | 0.1993 | 0.1475 | 0.0889 | 0.0690 | 0.0646 |
| **dK2P** | 0.2229 | 0.1963 | 0.0893 | - | - |
| **Site-Invariant-S** | 0.1274 | 0.1389 | 0.1018 | 0.0810 | 0.2087* |
| **Full-Invariant-S** | 0.1253 | 0.1527 | 0.0975 | 0.0695 | 0.1382* |
| **Sites-Attention-P** | 0.1269 | 0.1219 | 0.1060 | 0.0661 | 0.0716* |
| **Hybrid-Attention-SP** | **0.1163** | **0.1011** | **0.0856** | **0.0532** | 0.1299* |
| **Full-Attention-S** | 0.1664 | 0.1359 | 0.1222 | 0.0804 | 0.0826* |
| **Full-Attention-SP** | 0.1664 | 0.1020 | 0.0854 | 0.0547 | **0.0558*** |
| **Phyloformer** | - | - | - | **0.0506** | 0.0601 |
| **IQ-TREE** | 0.1732 | 0.1613 | 0.0791 | 0.0438 | 0.0453 |

**最佳性能**：Hybrid-Attention-SP在JC/K2P/HKY上最优；IQ-TREE在LG/LG+indel上最优

### 与基线方法对比
- **优于经典距离**：在大多数条件下，DNN方法显著优于dH、dJC和dK2P
- **接近IQ-TREE**：在复杂模型（HKY/LG）上，Hybrid-Attention-SP和Full-Attention-SP性能接近IQ-TREE
- **优于Phyloformer**：在内存效率上更优（表1），在LG模型上部分架构性能相当

### 消融实验结果
1. **架构复杂度与模型复杂度匹配**：
   - 简单模型（JC/K2P）：简单网络（Site-Invariant-S）表现更好
   - 复杂模型（HKY/LG）：混合/全注意力网络表现更好

2. **注意力头数量**：减少到4个以下时性能急剧下降，与理论结果一致

3. **隐藏通道数**：32-128通道为最优范围，过低或过高都会增加验证误差

4. **序列长度影响**：性能随序列长度增加而提升（图2），证明统计一致性

5. **Taxa数量影响**：RF距离随taxa数量增加而增加（图2），但相对性能保持稳定

6. **度量性质**：部分训练网络（如Sites-Attention-P）在未加正则化的情况下仍满足三角不等式

---

## 4. 关键结论和发现

### 主要发现
1. **信息共享的悖论**：令人惊讶的是，在taxa维度上不共享信息的dJC、Site-Invariant-S和Sites-Attention-P表现良好；而纯Taxa-Attention网络表现极差。这表明**仅需少量taxa间信息共享**即可，过多可能引入噪声

2. **混合架构的有效性**：Hybrid-Attention-SP（先taxa+site attention，后仅site attention）经常优于Full-Attention-SP，说明**早期层捕获taxa关系，深层应专注于site模式压缩**

3. **注意力机制的压缩效应**：对于i.i.d.数据，site attention起到正则化作用，将site模式压缩至输入的2%（DNA）或15%（蛋白）

4. **泛化能力**：训练网络在分布内数据上表现良好，但在**连接比对（concatenated alignments）**上转移性能差，无法匹配dJC或dK2P的准确性

### 方法局限性
1. **依赖模拟数据**：所有训练数据均为模拟生成，与真实基因组数据存在差距
2. **连接比对挑战**：对包含多个重组块（不同基因树）的真实基因组数据集处理不佳
3. **内存瓶颈**：自注意力对长序列的二次方依赖仍是扩展到数百万位点的主要障碍
4. **模型特异性**：当前为每个进化模型单独训练，实际应用需要统一模型

### 未来工作方向
1. **统一模型训练**：开发具有足够容量的单一模型，能够学习并区分多种进化模型
2. **连接比对扩展**：结合序列网络、混合网络或池化策略处理大规模连接比对
3. **内存优化**：采用局部注意力、分组查询注意力（Grouped-Query Attention）等技术降低内存占用
4. **损失函数改进**：探索基于四重奏一致性（quartet consistency）的对比学习损失，或优化NJ的ℓ∞边半径准则
5. **整合树构建过程**：将NJ算法整合到推理过程中，可能提高效率并减少长枝吸引（long-branch attraction）问题

---

## 核心贡献总结
该工作首次从理论上系统论证了**神经网络近似系统发育距离函数的可行性**，并设计了**可扩展、对称性保持的最小化架构**。关键创新在于**将度量学习理论与进化模型复杂性相匹配**，在计算效率和推断准确性之间取得了良好平衡，为开发通用、可迁移的系统发育学机器学习方法奠定了基础。

---

### 8. [Tensor Network Based Feature Learning Model](https://arxiv.org/abs/2512.02547)

**Authors**: Albert Saiapin, Kim Batselier  
**Category**: cs.LG  
**Published**: 2025-12-03  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2512.02547v1  

#### Abstract
Many approximations were suggested to circumvent the cubic complexity of kernel-based algorithms, allowing their application to large-scale datasets. One strategy is to consider the primal formulation of the learning problem by mapping the data to a higher-dimensional space using tensor-product stru...

#### AI Summary (by kimi-k2-thinking)
# Tensor Network Based Feature Learning Model 论文总结

## 1. 论文的主要贡献和创新点

### 解决了什么问题
- **核方法超参数选择效率瓶颈**：传统核方法（如SVM、GP）需要计算O(N³)的核矩阵，不适用于大规模数据(N≈10⁷-10⁹)
- **Cross-validation计算冗余**：现有tensor network核机器需对超参数θ进行cross-validation，需重复求解P次优化问题，计算成本高
- **特征映射与模型参数分离优化**：传统方法将特征超参数选择与模型参数训练割裂，未利用tensor结构特性

### 提出的新方法
**Feature Learning (FL)模型**：
- **可学习的特征组合**：将特征映射表示为P个不同超参数θ_p的tensor-product特征的线性组合
  $$f(\boldsymbol{x})=\left[\sum_{p=1}^{P}\lambda_{p}\boldsymbol{\phi}_{\theta_{p}}(\boldsymbol{x})\right]^{\top}\boldsymbol{w}$$
- **双重CPD结构**：同时用Canonical Polyadic Decomposition表示模型参数w（秩R）和特征映射（秩P）
- **统一优化框架**：将超参数搜索转化为对λ参数的正则化优化问题，与模型参数w联合训练

### 相比现有方法的优势
- **计算效率提升**：训练速度比cross-validation快**3-5倍**，复杂度从O(I²ᴰ[N+Iᴰ])降至O(EDNIR[P+IR])
- **预测性能相当或更优**：在多个数据集上MSE与cross-validation相当，在Yacht等数据集上显著更优
- **自动特征选择**：L1正则化使λ稀疏，自动识别重要特征尺度，减少后续计算
- **内存高效**：量化版本内存复杂度仅O(NRP)，适用于大规模高维数据

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 样本量N | 维度D | 类型 |
|--------|---------|-------|------|
| Airfoil | 1,502 | 5 | UCI回归 |
| Concrete | 1,030 | 8 | UCI回归 |
| Energy | 768 | 8 | UCI回归 |
| Wine | 6,497 | 11 | UCI回归 |
| Yacht | 308 | 6 | UCI回归 |
| **Airline** | **5,929,413** | **8** | **大规模** |

### 实验设置
- **数据预处理**：输入x缩放至单位超立方体，输出y标准化
- **特征映射**：量化Fourier features，每维I_d = I个基函数
- **模型结构**：CPD秩R，特征数P=8（Airline数据集P=6）
- **训练配置**：10个epoch，10次随机重启（Airline为5次）
- **超参数**：α=0.01, β=0.1，确保模型欠参数化(N > 参数总量)

### 评估指标
- **预测质量**：测试集Mean Squared Error (MSE)及标准差
- **训练效率**：总训练时间（秒）
- **对比基线**：量化CPD核机器 + 6-fold cross-validation (CV)

---

## 3. 主要实验结果和性能指标

### 正则化方法对比（表1）
| 正则化类型 | 平均MSE | 平均训练时间 | 特点 |
|------------|---------|--------------|------|
| **L1** | **最优** (0.003-0.692) | **最快** (0.1-27.8s) | **产生稀疏λ，自动特征选择** |
| L2 | 略高0.5-3% | 慢10-20% | 解密集，需调β |
| Fixed Norm | 相当 | 中等 | **无需调β**，但解始终密集 |

**关键发现**：L1正则化在MSE和速度上综合最优，约30-50%的λ_p被稀疏化为0

### FL vs Cross-Validation对比（表2）
| 数据集 | MSE (FL) | MSE (CV) | 时间 (FL) | 时间 (CV) | **加速比** |
|--------|----------|----------|-----------|-----------|------------|
| Airfoil | 0.184±0.02 | 0.223±0.02 | 3.0s | 23s | **7.7×** |
| Energy | 0.003±0.0 | 0.003±0.0 | 0.91s | 5.5s | **6.0×** |
| Yacht | **0.112±0.02** | 0.358±0.06 | 0.149s | 0.615s | **4.1×** |
| Concrete | 0.139±0.03 | 0.118±0.02 | 1.2s | 8.8s | **7.3×** |
| Wine | 0.692±0.07 | 0.652±0.04 | 33s | 152s | **4.6×** |
| **Airline** | 0.804±0.0 | 0.779±0.0 | **15,159s** | **56,590s** | **3.7×** |

**性能分析**：
- **速度**：在所有数据集上FL均显著更快，**3.7-7.7倍加速**
- **精度**：4/6数据集性能相当（std区间重叠），Yacht上FL**显著更优**（MSE降低69%）
- **可扩展性**：在百万级样本的Airline数据集上，FL仅需4.2小时 vs CV的15.7小时

### 消融实验（图2）
- **P（特征数）影响**：随P增大，FL训练时间增长**更缓慢**（斜率更低）
- **鲁棒性**：FL的MSE标准差区域与CV高度重叠，证明**稳定性相当**
- **表达性**：FL的线性组合特征比CV的单一特征更具表达能力

---

## 4. 关键结论和发现

### 主要发现
1. **效率与精度兼得**：FL模型通过联合优化特征超参数和模型参数，在保持预测质量的同时实现数量级的加速
2. **结构优势显著**：双重CPD结构使ALS算法能将非凸问题分解为一系列线性子问题，大幅降低计算复杂度
3. **自动特征尺度学习**：λ参数有效捕捉不同θ_p的重要性，L1正则化实现**隐式特征选择**
4. **大规模可扩展**：在5.9M样本的Airline数据集上验证，FL的O(EDNKR[P+R])复杂度线性扩展

### 方法局限性
- **超参数仍存在**：需手动设置α（模型正则化）和β（λ正则化），Fixed Norm可缓解但无法完全消除
- **非负约束权衡**：L1(P)强制λ≥0会轻微降低部分数据集性能（如Concrete MSE从0.139升至0.176）
- **CPD秩选择**：R的选择仍依赖经验，过高可能导致过拟合
- **理论保证不足**：仅验证局部收敛性，全局最优性未理论证明

### 未来工作方向
1. **并行化**：利用模型求和结构开发并行FL算法
2. **概率化**：构建概率FL模型，为分类和回归提供不确定性估计
3. **自适应秩**：开发动态调整CPD秩R的机制
4. **更广应用**：探索除Fourier外的特征映射（如多项式、小波）

---

**核心贡献总结**：该工作首次将tensor network的CPD结构同时应用于特征映射和模型参数，开创了**超参数与参数联合学习**的新范式，为大规模核方法提供了高效、可扩展的解决方案。

---

### 9. [FiMMIA: scaling semantic perturbation-based membership inference across modalities](https://arxiv.org/abs/2512.02786)

**Authors**: Anton Emelyanov, Sergei Kudriashov, Alena Fenogenova  
**Category**: cs.LG  
**Published**: 2025-12-03  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2512.02786v1  

#### Abstract
Membership Inference Attacks (MIAs) aim to determine whether a specific data point was included in the training set of a target model. Although there are have been numerous methods developed for detecting data contamination in large language models (LLMs), their performance on multimodal LLMs (MLLMs...

#### AI Summary (by kimi-k2-thinking)
# FiMMIA: 跨模态语义扰动驱动的成员推断攻击

## 1. 主要贡献和创新点

### 解决的问题
- **多模态大语言模型(MLLMs)的数据污染检测难题**：现有LLMs的MIA方法无法直接应用于MLLMs，主要障碍包括：
  - 多模态组件适配引入的不稳定性
  - 跨模态输入的分布偏移
  - 缺乏模态相关token的目标标签
  - 多阶段训练流程增加推断复杂度

### 提出的新方法
- **FiMMIA框架**：首个模块化多模态MIA框架，支持图像、视频、音频和文本模态
- **语义扰动扩展**：将基于扰动的MIA方法推广到多模态场景，通过分析目标模型在扰动输入上的行为差异来捕捉成员/非成员数据的分布差异
- **分布偏移检测基线**：扩展Das等人(2024)的工作，提出不依赖目标模型信号的盲统计检测方法

### 相比现有方法的优势
- **跨模态通用性**：统一处理图像、视频、音频和文本，而现有方法多局限于单一模态
- **可转移性**：训练的攻击模型在不同模型家族间表现出良好的迁移能力
- **鲁棒性**：通过z-score归一化和邻居样本差异特征，降低对绝对统计值的依赖
- **实用性**：提供完整的开源管道和预训练模型，支持即插即用

---

## 2. 核心实验方法和设置

### 数据集
- **主要评估**：MERA基准测试（俄语多模态数据集），包含18个音频、视频和图像数据集
- **跨语言迁移**：英语数据集
  - ScienceQA（多选题）
  - MMStar（多选题）
  - COCO-Caption2017（图像描述）
- **分布偏移评估**：测试了WikiMIA、VL-MIA、LAION-MI等现有MIA基准

### 实验设置
- **目标模型**：9个公开MLLMs（3B-12B参数）
  - Qwen2-VL系列（3B/7B）
  - Gemma-3系列（4B/12B）
  - LLaVA-NeXT系列（8B）
  - Qwen-Audio系列（7B）
- **训练配置**：
  - 使用LoRA微调场景
  - 每个样本生成K=24个扰动邻居
  - 采用5折交叉验证
  - 测试集占原始数据10%

### 评估指标
- **AUC-ROC**：二分类（泄露 vs 干净）性能
- **TPR@low FPR**：FPR=5%时的真阳性率
- **跨模型转移**：训练模型(M_origin)与测试模型(M_test)不一致时的性能

### 基线方法
- **盲统计基线**：提取SIFT、LBP、MFCC等手工特征，用浅层ML模型分类
- **MMDETECT**：Song等人(2025)的Δ-score方法
- **现有SOTA**：与WikiMIA、VL-MIA等报告的最佳结果对比

---

## 3. 主要实验结果和性能指标

### 整体性能（AUC-ROC）
| 模态 | 平均AUC-ROC | 最佳单模型性能 |
|------|-------------|----------------|
| 图像 | 88.66% | 99.7% (Gemma-3-12b-it) |
| 视频 | 88.39% | 100.0% (Qwen2.5-VL-7B) |
| 音频 | 81.25% | 100.0% (Qwen-Audio-Chat) |

### 跨模型转移性能
- **同一家族内转移**：性能下降有限（通常<10%）
  - Qwen2.5-VL-3B → Qwen2.5-VL-7B: 88.0% AUC
  - Qwen2-VL-7B → Qwen2.5-VL-7B: 80.5% AUC
- **跨家族转移**：性能下降较明显但仍有效
  - Qwen → LLaVA: 78.0-96.6% AUC
  - Qwen → Gemma: 65.8-95.4% AUC
  - 最低性能仍>65%，显示方法具有普适性

### 与基线对比
- **分布偏移检测**：在多数现有MIA基准上，盲统计基线 outperform 报告的最佳攻击结果
  - WikiMIA-hard: 57.7% vs 64.0%（报告值）
  - VL-MIA-Flickr: 99.1% vs 94.2%（报告值）
  - LAION-MI: 仅1.11% TPR@1FPR，显示无明显分布偏移

- **跨语言迁移**：FiMMIA在英语数据集上表现与MMDETECT一致
  - COCO: 0.00-0.58% 泄露样本检测率
  - MMStar: 0.00-0.13% 泄露样本检测率
  - ScienceQA: 0.00-0.21% 泄露样本检测率

### TPR@FPR=5%结果
- **图像模态**：最佳达99.6%（Gemma-3-12b-it自测）
- **视频模态**：最佳达100%（同模型自测）
- **音频模态**：最佳达100%（Qwen-Audio-Chat自测）
- **跨模型转移**：TPR下降幅度大于AUC，最低至6.0%（LLaVA-NeXT-Video → Qwen2.5-VL-3B）

---

## 4. 关键结论和发现

### 主要发现
1. **分布偏移是现有MIA基准的根本缺陷**：大多数文本和图像MIA数据集存在严重分布偏移，导致盲统计方法即可达到高AUC，使其不适合评估真实MIA性能。LAION-MI是少数无明显偏移的基准。

2. **语义扰动对MLLMs高度有效**：即使模态数据保持不变（仅扰动文本），通过分析模型在文本扰动上的损失差异，仍能有效识别训练成员。这表明MLLMs的跨模态对齐机制会记忆特定文本-模态对。

3. **跨模型可转移性存在家族偏差**：攻击模型在同一家族内迁移效果最佳，跨架构迁移时性能下降但仍有实用价值（AUC>65%），证明方法捕捉了与模型无关的通用记忆特征。

4. **跨语言泛化能力**：仅在俄语数据上训练的FiMMIA可直接应用于英语数据集，结果与专用英语方法MMDETECT一致，验证了框架的语言无关性设计。

### 方法局限性
- **计算成本**：单数据集单GPU推理约需10小时，复杂度为O(|D|N(M+E+G))
- **灰盒假设**：需要访问模型logits计算损失，严格黑盒场景不适用
- **微调场景限制**：仅在LoRA微调场景验证，预训练和全微调效果未知
- **随机性影响**：硬件-软件栈（GPU、CUDA、PyTorch版本等）引入非确定性，尽管影响有限

### 未来工作方向
- 扩展至预训练和全微调场景
- 实现黑盒API场景（仅top-k logits）的支持
- 优化计算效率，降低推理成本
- 探索模态扰动（不仅限于文本）的效果
- 构建无分布偏移的标准化多模态MIA基准

---

**核心贡献总结**：FiMMIA通过将语义扰动MIA扩展到多模态领域，揭示了现有基准的分布偏移问题，并提供了一个可扩展、可转移的实用框架，为MLLMs的数据隐私审计和基准可靠性验证提供了有效工具。

---

### 10. [Beyond Playtesting: A Generative Multi-Agent Simulation System for Massively Multiplayer Online Games](https://arxiv.org/abs/2512.02358)

**Authors**: Ran Zhang, Kun Ouyang, Tiancheng Ma, Yida Yang, Dong Fang  
**Category**: cs.AI  
**Published**: 2025-12-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.02358v1  

#### Abstract
Optimizing numerical systems and mechanism design is crucial for enhancing player experience in Massively Multiplayer Online (MMO) games. Traditional optimization approaches rely on large-scale online experiments or parameter tuning over predefined statistical models, which are costly, time-consumin...

#### AI Summary (by kimi-k2-thinking)
# Beyond Playtesting: A Generative Multi-Agent Simulation System for Massively Multiplayer Online Games

## 1. 主要贡献和创新点

### 解决的问题
- **传统优化方法的局限性**：MMO游戏数值系统和机制设计优化依赖大规模在线实验或预定义统计模型，存在**高时间成本**（需数周/月观察）、**高机会成本**（错误调整可能导致不可逆的经济系统和玩家流失）以及**测试限制**（重大机制变更无法通过小规模A/B测试验证）
- **现有模拟系统的不足**：简化离线模拟系统保真度有限，无法准确模拟真实玩家的推理过程和对干预措施的动态反应，且作为"黑盒"缺乏微观玩家层面的可解释洞察

### 提出的新方法
- **生成式多智能体模拟框架**：首次构建可扩展的LLM驱动MMO游戏模拟系统，通过**Supervised Fine-Tuning (SFT)** 和 **Reinforcement Learning (RL)** 在大规模真实玩家行为数据上适配LLM
- **三阶段微调流程**：
  1. **Vocabulary Expansion**：扩展tokenizer并训练游戏特定token嵌入（如"AWM"等术语）
  2. **Action Planning SFT**：基于历史行为、环境反馈和玩家画像预测下一步行动（offline/battle/buy/sell）
  3. **RL Enhancement**：使用GRPO强化学习增强推理能力和泛化性
- **数据驱动环境模型**：基于真实游戏日志训练Battle Server，重建动态游戏系统（战斗、NPC商店、黑市交易）

### 相比现有方法的优势
- **高保真度**：通过真实行为数据微调，智能体能复现人类玩家的感知、推理和决策模式
- **可解释性**：提供详细的智能体行为和推理分析，超越传统统计模型的"黑盒"局限
- **成本效益**：离线模拟避免昂贵的在线实验风险，支持快速迭代
- **系统性模拟**：全面重建游戏生态系统，捕捉玩家行为间的相互关联后果（如战斗结果影响购买决策）

---

## 2. 核心实验方法和设置

### 数据集
- **行为数据**：数百万真实玩家游戏记录（登录/登出、战斗、购买、社交互动），用于构建玩家画像
- **战斗数据**：2025S1赛季连续匹配日志（玩家参与35-40场比赛），覆盖多个赛季
- **验证数据**：2025S2赛季数据，严格确保无数据泄露

### 实验设置
- **基础模型**：Qwen2.5-1.5B
- **微调配置**：LoRA适配器（rank=16, α=0.2）
- **RL算法**：GRPO（Generalized Reward Policy Optimization）
- **玩家分群**：基于十多项游戏特征聚类为5类代表画像：
  1. Stable Development Players
  2. Novice Players
  3. Wealth-Accumulating Elite Players
  4. Casual Players
  5. High-skill Players

### 评估指标
- **Player Agent**：下一步行动预测准确率（四分类：offline/battle/buy/sell）
- **Battle Server**：胜负预测准确率、每场收入回归预测误差
- **系统级评估**：财富分布、排名分布、资源消耗、活跃度等宏观统计一致性
- **干预评估**：行为分布变化率（如非正式交易比例）

### 基线方法
- **DeepSeek-V3**（未微调状态）
- 消融对比：仅SFT vs SFT+用户画像 vs SFT+RL

---

## 3. 主要实验结果和性能指标

### Player Agent性能
- **整体准确率提升**：在10,000条单日游戏轨迹上评估
  - DeepSeek-V3基线：基准性能
  - **+8.34%**：仅行为数据SFT带来的提升
  - **+1.85%**：加入用户画像信息的额外增益
  - **总计+10.19%**：三阶段完整流程的改进效果
- **数据分布**：行动类别天然不平衡（battle > offline），模型在imbalanced数据上表现稳健

### Battle Server预测精度
- **胜负预测**：对5类玩家群体的N-th比赛胜率预测与2025S2真实数据高度对齐（允许固定偏移）
  - **最佳表现**：Wealth-Accumulating Elite Players 和 Stable Development Players
  - **较大波动**：Novice Players 和 Casual Players（行为随机性更高）
- **收入预测**：每场收入预测同样与真实数据对齐，精英玩家群体预测最准确

### 干预案例研究结果
- **场景**：引入官方黑市交易系统（替代原有非正式交易）
- **效果量化**：
  - **非正式交易比例**：从 **27.4% → 1.5%**（下降94.5%）
  - **黑市采用率**：绝大多数智能体转向新渠道
  - **残留行为**：仅1.5%因干预前习惯性行为继续使用旧方式
- **因果效应验证**：成功复现真实世界中干预措施的玩家行为响应模式

---

## 4. 关键结论和发现

### 主要发现
1. **高保真模拟**：通过SFT和RL适配的LLM智能体在宏观统计分布和微观个体行为层面均与真实玩家表现出强一致性
2. **干预预测能力**：系统能忠实再现游戏机制变更的因果效应，为设计决策提供可靠预测
3. **可扩展架构**：基于Python异步框架和MQTT消息队列，支持大规模智能体并发（数百智能体模拟数周游戏时间）
4. **画像驱动差异**：不同玩家群体对相同机制的反应存在显著差异，系统能捕捉这些异质性

### 方法局限性
- **新手/休闲玩家预测**：Novice和Casual玩家群体因行为随机性大，预测误差相对较高
- **数据依赖性**：模型性能受限于真实数据的质量和覆盖范围，可能无法很好模拟罕见边缘案例
- **计算成本**：LLM推理和RL训练需要 substantial 计算资源，尽管低于在线实验成本
- **确定性系统假设**：NPC商店和黑市被建模为确定性系统，可能忽略微观随机性

### 未来工作方向
- **增强鲁棒性**：改进对低活跃度/高随机性玩家群体的建模
- **扩展机制覆盖**：纳入更多游戏系统（如社交、公会、任务链）
- **实时校准**：开发在线适应机制，使模拟系统能持续学习新数据
- **多游戏验证**：在更多MMO品类（非仅extraction shooter）验证框架通用性
- **优化效率**：探索模型压缩和蒸馏技术，降低大规模模拟的计算开销

---

### 11. [Dynamic Configuration of On-Street Parking Spaces using Multi Agent Reinforcement Learning](https://arxiv.org/abs/2512.02406)

**Authors**: Oshada Jayasinghe, Farhana Choudhury, Egemen Tanin, Shanika Karunasekera  
**Category**: cs.LG  
**Published**: 2025-12-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.02406v1  

#### Abstract
With increased travelling needs more than ever, traffic congestion has become a major concern in most urban areas. Allocating spaces for on-street parking, further hinders traffic flow, by limiting the effective road width available for driving. With the advancement of vehicle-to-infrastructure conn...

#### AI Summary (by kimi-k2-thinking)
# 论文核心结论与实验结果总结

## 1. 主要贡献和创新点

### 解决的问题
论文针对**城市交通拥堵**问题，特别是**路边停车对交通流量的负面影响**（如减少有效车道宽度、在交叉口造成队列不平衡）提出解决方案。传统静态Clearways方案无法适应动态交通状况，而现有停车优化研究多从用户视角出发（寻找车位），缺乏从基础设施视角动态配置停车空间的机制。

### 提出的新方法
- **双层多智能体强化学习框架（Two-layer MARL）**：
  - **车道级智能体（Lane Level RL Agents）**：部署在每个车道，决定清除或保留的停车位数量
  - **区块级智能体（Block Level Agents）**：监控区块内停车供需平衡，协调车道级智能体的动作
  
- **新型Deep Q-learning架构**：
  - 结合**LSTM网络**捕获本车道交通拥堵的时间序列变化
  - 结合**Graph Attention Networks (GAT)** 捕获相邻车道的时空相关性
  - 采用**集中训练、分布执行（Centralized Training with Decentralized Execution）** 范式

### 相比现有方法的优势
- **主动预测性**：通过时空相关性建模，能预判未来交通状况并提前采取行动（清除关键停车位）
- **细粒度控制**：以单个停车位为配置单位，比Clearways的整段禁停更灵活
- **可扩展性强**：分层架构天然适用于大规模路网
- **多目标平衡**：同时优化交通流量（最小化旅行时间损失）和停车便利性（最小化步行距离）

---

## 2. 核心实验方法和设置

### 数据集
- **真实数据**：墨尔本市中心3.5km×2km区域
  - 15个信号交叉口，38个路段，**3,042个路边停车位**
  - 71,217辆车的全天轨迹数据（6am-9pm）
  - 使用SCATS感应线圈数据和停车传感器数据
  
- **合成数据**：网格网络（3×3至7×7）
  - 用于**敏感性分析**，测试不同交通密度、停车概率、停车时长下的鲁棒性

### 实验设置
- **仿真平台**：SUMO微观交通仿真器 + TraCI Python API实时交互
- **训练框架**：PyTorch，Adam优化器，MSE损失函数
- **网络参数**：
  - 全连接层：2→32维
  - LSTM：2层，32维隐藏状态，序列长度10
  - GAT：2个注意力头，64维隐藏状态
  - Q值预测网络：3层FCN（96→128→3）

### 评估指标
- **平均时间损失**（Average Time Loss, $t_{loss}$）：实际旅行时间与自由流时间差
- **时间损失减少百分比**（$t_{loss}\%$）：相对于No-PA基线的改善幅度
- **平均步行距离**（$d_{walk}$）：目的地与实际停车位置的距离
- **车辆排放**：CO₂、CO、HC、PMx、NOx

### 基线方法
- **No-PA**：无停车限制
- **C-PA**：Clearways（高峰时段整段禁停）
- **S-PA**：静态清除交叉口附近固定数量停车位
- **D-PA变体**：PPO、A2C、Double DQN、Dueling DQN

---

## 3. 主要实验结果和性能指标

### 真实数据核心结果（停车概率0.4场景）
| 方法 | $t_{loss}$ (秒) | $t_{loss}\%$ 减少 | $d_{walk}$ (米) |
|------|----------------|-------------------|-----------------|
| No-PA | 232.17 | - | 2.12 |
| S-PA | 200.21 | 13.77% | 2.47 |
| C-PA | 198.55 | 14.48% | 70.70 |
| **D-PA (Ours)** | **121.22** | **47.79%** | **3.52** |

**关键发现**：
- **旅行时间损失降低47.79%**，远超其他方法（最佳基线仅14.48%）
- **步行距离仅增加1.4米**（从2.12米到3.52米），而Clearways增加68.58米
- 在所有停车概率设置下（0.1-0.4）均保持最优性能

### 敏感性分析结果
- **交通密度**：车辆插入率从60 veh/s增至110 veh/s时，D-PA优势更显著（时间损失差值扩大）
- **停车概率**：停车概率0.4时，C-PA性能急剧恶化，而D-PA保持稳定
- **停车时长**：从600秒增至3600秒，D-PA始终维持最低时间损失
- **网格规模**：从3×3到7×7网格，D-PA可扩展性良好，性能优势不变

### 消融实验结果
| 架构 | 停车概率0.4时的$t_{loss}\%$减少 |
|------|-------------------------------|
| DQN (vanilla) | 32.93% |
| DQN + LSTM | 38.52% |
| **DQN + LSTM + GAT** | **47.79%** |

**结论**：LSTM捕获时序信息带来**5.59%**提升，GAT捕获空间相关性带来**9.27%**额外提升，证明**时空建模对性能至关重要**。

### 环境效益
- **排放减少**：CO₂、CO、HC、PMx、NOx最高减少**40%**
- **全天候改善**：图6显示在早晚高峰时段（8-9am, 5-6pm）效果最显著

---

## 4. 关键结论和发现

### 主要发现
1. **动态配置显著优于静态方案**：相比Clearways和静态清除，动态配置在交通流量和停车便利性间取得更好平衡
2. **时空相关性建模是核心**：LSTM+GAT架构使智能体能预判拥堵传播，提前清除关键车位
3. **双层架构有效性**：区块级智能体成功防止车道级智能体过度清除车位，确保停车可用性
4. **V2I技术使能**：依赖车路协同技术实时传递停车限制信息，可通过手机App或车载系统实现

### 方法局限性
- **动作不可逆性**：一旦允许停车，车辆在停放期间无法清除该车位，要求策略高度前瞻性
- **监管挑战**：需符合交通法规框架，获得社会接受度
- **部署成本**：需V2I基础设施支持（尽管比电子路牌成本低）

### 未来工作方向
- 扩展至**自动驾驶和混合交通场景**
- 探索更复杂的动作空间（如动态定价、分时共享）
- 研究长期学习稳定性和跨城市迁移能力

---

**总结**：该论文首次将路边停车空间配置建模为MARL问题，通过创新的DQN+LSTM+GAT架构和双层协调机制，在真实城市数据中实现**近50%的旅行时间减少**，同时仅增加不到2米步行距离，为智能交通系统提供了可扩展、高效的动态停车管理方案。

---

### 12. [In-Context Distillation with Self-Consistency Cascades: A Simple, Training-Free Way to Reduce LLM Agent Costs](https://arxiv.org/abs/2512.02543)

**Authors**: Vishnu Sarukkai, Asanshay Gupta, James Hong, Micha\"el Gharbi, Kayvon Fatahalian  
**Category**: cs.LG  
**Published**: 2025-12-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.02543v1  

#### Abstract
The world currently has an abundance of ideas for how to use new LLM agents, and developers seek to rapidly prototype and test new agentic designs. However, executing agents at scale using high-capacity LLMs incurs high inference costs. We propose a simple method for reducing LLM agent inference cos...

#### AI Summary (by kimi-k2-thinking)
# In-Context Distillation with Self-Consistency Cascades 论文总结

## 1. 主要贡献和创新点

### 解决的问题
- **LLM Agent规模化部署的高推理成本**：执行新型LLM-based agents在规模化应用时面临高昂的推理成本
- **传统成本降低方法的开发摩擦**：模型蒸馏需要微调（训练周期长、超参数调优），提示工程需要大量人工试错，均阻碍快速原型开发

### 提出的新方法
- **In-Context Distillation（上下文蒸馏）**：将知识蒸馏思想适配到in-context learning场景
  - 在agent每个决策步骤检索相关教师演示（包含推理轨迹）
  - 将演示作为in-context examples提供给冻结的学生模型
  - 使学生无需参数更新即可动态模仿教师行为
  
- **Self-Consistency Cascades（自一致性级联）**：采用自一致性作为内省信号判断何时信任学生
  - 对相同检索示例采样多个学生输出
  - 样本一致则执行学生动作，不一致则回退到教师
  - 在单个agent步骤粒度上实现自适应路由

### 相比现有方法的优势
- **无需训练**：完全避免模型权重更新，消除计算开销和ML专业知识需求
- **敏捷开发**：保持使用冻结模型的开发速度，支持快速迭代
- **成本效益显著**：在ALFWorld上实现**2.5倍成本降低**（$0.059 → $0.024/episode），在AppWorld上实现**2倍成本降低**
- **数据高效**：仅需几百个教师演示即可有效，演示成本在843（ALFWorld）和207（AppWorld）个episode后摊销
- **性能保持**：在ALFWorld上达到甚至超过教师准确率（96% vs 89%），在AppWorld上恢复79%教师准确率

---

## 2. 核心实验方法和设置

### 数据集
| 数据集 | 类型 | 演示任务数 | 测试任务数 | 特点 |
|--------|------|------------|------------|------|
| **ALFWorld** | 具身推理 | 500（train split） | 134（eval-out-of-distribution） | 离散动作空间，多步规划问题 |
| **AppWorld** | API工作流自动化 | 147（train/val splits） | 168（test-normal） | 真实业务工作流，Python代码动作空间 |

### 实验设置
- **模型配置**：
  - Teacher：Claude Sonnet 4.5（高能力）
  - Student：GPT-4.1-mini（主要），Llama-3.3-70B（验证通用性）
  - Embedding模型：MiniLM-L6-v2（用于检索）
  
- **关键参数**：
  - 检索示例数k：ALFWorld=6，AppWorld=3
  - 自一致性采样数N=3
  - Temperature=0.1（默认）
  - 动作空间：ALFWorld使用预定义动作，AppWorld使用Python代码

### 评估指标
- **任务成功率**：episode成功完成的比例
- **推理成本**：基于API定价计算的总token成本（输入+输出）
  - GPT-4.1-mini: $0.40/M输入, $1.60/M输出
  - Claude Sonnet 4.5: $3.00/M输入, $15.00/M输出
- **相对成本**：相对于教师基线的归一化成本

### 基线方法对比
1. **Teacher**：仅使用教师模型（准确率上限）
2. **Student (ZS)**：零样本学生（成本下限，准确率基线）
3. **Student (IC)**：仅上下文蒸馏（测试蒸馏单独效果）
4. **Student (IC+Cascade)**：完整方法（主要贡献）
5. **Random Mix**：固定比例混合（控制变量）
6. **Student (Cascade only)**：仅级联无示例（隔离级联价值）
7. **GPT-4.1 (ZS)**：中等规模模型零样本
8. **SOTA系统**：如IBM's CuGA（73.2%准确率）

---

## 3. 主要实验结果和性能指标

### ALFWorld结果
| 方法 | 准确率 | 相对成本 | 绝对成本 | 关键发现 |
|------|--------|----------|----------|----------|
| Teacher | 0.89 | 1.0 | $0.059 | 教师基线 |
| Student (ZS) | 0.18 | 0.31 | $0.018 | 学生单独性能差 |
| Student (IC) | **0.87** | 0.43 | $0.026 | **97%教师准确率，43%成本** |
| Student (IC+Cascade) | **0.96** | **0.41** | **$0.024** | **超过教师，2.5倍成本降低** |

**成本效益**：演示成本$29.50，在**843 episodes后摊销**，1M episodes节省 **$34,900**

### AppWorld结果
| 方法 | 准确率 | 相对成本 | 绝对成本 | 关键发现 |
|------|--------|----------|----------|----------|
| Teacher | 0.83 | 1.0 | $0.589 | 教师基线 |
| Student (ZS) | 0.28 | 0.09 | $0.051 | 零样本性能低 |
| Student (IC) | 0.55 | 0.15 | $0.088 | **准确率提升96%，成本15%** |
| Student (IC+Cascade) | **0.66** | **0.29** | **$0.171** | **2倍成本降低，79%教师性能** |

**成本效益**：演示成本$86.73，在**207 episodes后摊销**，1M episodes节省 **$419,000**

### 消融实验结果

#### 检索示例数量（k）的影响
- **ALFWorld**：k=1→4时准确率从0.75→0.86（+13.8%），k>6后边际收益递减
- **AppWorld**：k=5时达到峰值0.57，k>5后无一致收益
- **结论**：k=4-6为最佳性价比区间

#### 教师数据库规模影响
- **ALFWorld**：100 demonstrations → 0.836准确率（94%教师性能）；500 → 0.87（98%教师性能）
- **AppWorld**：147 demonstrations → 0.548准确率（67%教师性能）
- **结论**：上下文蒸馏数据效率高，小规模数据库即可显著超越零样本

#### 检索粒度对比
| 方法 | ALFWorld准确率 | ALFWorld成本 | AppWorld准确率 | AppWorld成本 |
|------|----------------|--------------|----------------|--------------|
| Per-step retrieval | 0.87 | **0.43** | 0.55 | **0.15** |
| Single retrieval | 0.87 | 0.54 (+26%) | 0.54 | 0.24 (+60%) |

**结论**：每步检索在保持准确率的同时显著降低成本（避免包含完整轨迹的冗余token）

#### 难度感知路由（AppWorld）
| 难度等级 | Teacher | Student (IC+Cascade) | Student (ZS) | 任务数 |
|----------|---------|----------------------|--------------|--------|
| 1（简单） | 0.96 | **0.91** (95%恢复) | 0.51 | 57 |
| 2（中等） | 0.85 | **0.70** (82%恢复) | 0.29 | 48 |
| 3（困难） | 0.71 | **0.43** (61%恢复) | 0.06 | 63 |

**结论**：方法在简单任务上接近教师，困难任务上差距扩大

---

## 4. 关键结论和发现

### 主要发现
1. **技术互补性**：In-context distillation和self-consistency cascades是互补技术，联合使用建立新的Pareto前沿
2. **上下文蒸馏的有效性**：单独使用即可将学生性能从0.18→0.87（ALFWorld），成本仅43%，证明教师知识可通过in-context learning有效转移
3. **级联的智能路由**：自一致性作为内省信号，仅在3.9%（ALFWorld）和22.2%（AppWorld）步骤触发教师查询，实现精准回退
4. **跨模型通用性**：方法在GPT-4.1-mini和Llama-3.3-70B上均有效，不依赖特定API
5. **与SOTA竞争力**：在AppWorld上达到65.5%准确率，接近IBM CuGA（73.2%），但无需复杂编排基础设施

### 方法局限性
1. **困难任务性能下降**：在AppWorld难度3任务上仅恢复61%教师性能（0.43 vs 0.71），复杂推理仍是挑战
2. **检索质量依赖**：当测试任务与演示任务差异大时，检索示例相关性低，学生性能下降
3. **Upfront演示成本**：需预先收集教师演示，尽管摊销快但对小规模部署不友好
4. **动作空间假设**：在API任务中需辅助验证器判断语义等价性，增加系统复杂性
5. **未优化KV缓存**：成本计算未考虑上下文缓存优化，实际成本可能更低

### 未来工作方向
1. **扩大数据库规模**：在AppWorld上探索>147 demonstrations的效果，进一步缩小准确率差距
2. **缓存感知成本优化**：实现KV缓存减少重复token成本
3. **混合路由策略**：结合自一致性与任务难度元数据，实现更精细的成本-准确率权衡
4. **多模态扩展**：将方法扩展到视觉等多模态agent场景
5. **在线演示更新**：探索动态更新教师数据库以适应分布变化（当前保持固定）

### 实践建议
- **快速原型开发**：优先使用此方法，避免微调开销
- **生产系统选择**：稳定高流量场景可考虑微调，本方法适合敏捷迭代和专用部署
- **参数配置**：k=4-6为通用最佳选择，N=3采样足以检测不确定性
- **成本敏感应用**：在AppWorld上仅需207 episodes即可回本，适合大规模批处理任务

---

### 13. [Adapting Tensor Kernel Machines to Enable Efficient Transfer Learning for Seizure Detection](https://arxiv.org/abs/2512.02626)

**Authors**: Seline J. S. de Rooij, Borb\'ala Hunyadi  
**Category**: cs.LG  
**Published**: 2025-12-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.02626v1  

#### Abstract
Transfer learning aims to optimize performance in a target task by learning from a related source problem. In this work, we propose an efficient transfer learning method using a tensor kernel machine. Our method takes inspiration from the adaptive SVM and hence transfers 'knowledge' from the source ...

#### AI Summary (by kimi-k2-thinking)
# Adapting Tensor Kernel Machines to Enable Efficient Transfer Learning for Seizure Detection - 核心总结

## 1. 主要贡献和创新点

### 解决的问题
- **医疗数据稀缺性**：癫痫检测中患者间变异性大，但患者特异性数据收集和标注成本高昂
- **传统迁移学习方法的效率瓶颈**：Adaptive SVM等核方法在适应过程中会增加支持向量，导致模型尺寸膨胀（O(N^s+N^t)），不适合资源受限的可穿戴设备
- **隐私顾虑**：传统方法需要存储源域训练数据（支持向量），存在隐私泄露风险

### 提出的新方法
**Adaptive Tensor Kernel Machine (Adapt-TKM)**：一种基于张量核机（Tensor Kernel Machine）的高效迁移学习方法
- 通过**正则化项**而非增加参数实现知识迁移：最小化 ||W - W^s||_F^2（适应模型与源模型权重间的Frobenius距离）
- 在**原始空间（primal domain）**优化，使用低秩张量网络（Canonical Polyadic Decomposition, CPD）压缩模型参数
- 采用块坐标下降法（block coordinate descent）高效求解

### 相比现有方法的优势
| 维度 | Adapt-TKM | Adaptive SVM |
|------|-----------|--------------|
| **模型参数** | ~100倍更少（1.99×10^4 vs 1.36×10^6） | 随支持向量线性增长 |
| **推理速度** | ~100倍更快（3.9×10^-5s vs 3.3×10^-3s） | 需计算所有支持向量核函数 |
| **模型尺寸** | 适应后不增加 | 适应后增加（源+目标支持向量） |
| **隐私保护** | 不存储源域数据 | 需存储源域支持向量 |
| **理论优势** | 在原始空间学习，避免对偶空间的O(N^2)复杂度 | 对偶空间优化，核矩阵计算昂贵 |

---

## 2. 核心实验方法和设置

### 数据集
- **SeizeIT1**：耳后EEG数据集，包含医院术前评估期间采集的多天监测数据
  - 患者：15名（发作≥5次）
  - 发作类型：主要为局灶性意识障碍发作（FIA, 89%），多数源于（额）颞叶（91%）
  - 通道：耳后EEG（每侧2个电极）的交叉头部通道和单侧横向通道
  - 标注：结合神经科医生和工程师的标注，确保发作持续时间≥10秒

### 实验设置
- **模型类型对比**：
  - **PI（Patient-Independent）**：留一患者交叉验证训练通用模型
  - **PS（Patient-Specific）**：仅使用患者特异性数据训练
  - **PA（Patient-Adapted）**：用Adapt-TKM或Adapt-SVM微调PI模型
- **数据划分**：
  - PI模型：留一患者交叉验证，仅使用耳后通道可见的发作训练
  - PA/PS模型：留一发作交叉验证，训练集随机采样非发作数据至1:10（发作:非发作）比例
- **预处理**：1-25Hz带通滤波，2秒分段（训练：发作段90%重叠，非发作段无重叠；测试：50%重叠），RMS幅度筛选（13-150μV）

### 特征提取
- **21个特征/通道**：时域（零交叉、峰度等）、频域（δ,θ,α,β,HF波段功率）、熵特征（谱熵、样本熵、Shannon熵）
- **变换**：Yeo-Johnson变换改善正态性，缩放至[-0.5, 0.5]

### 评估指标
- **事件级指标**（any-overlap方法）：
  - **Sensitivity**：发作检测召回率
  - **Precision**：精确率
  - **F1-score**：综合性能
  - **FA/24hr**：每日误报次数
- **后处理**：仅当连续10个片段中≥8个被分类为发作时才触发警报（最小发作持续时间10秒）

### 基线方法
- SVM-PI/PS/PA（线性核和RBF核）
- TKRR-PI/PS（无迁移的张量核岭回归）
- Adapt-SVM（传统自适应SVM）

---

## 3. 主要实验结果和性能指标

### 整体性能对比（表III）
| Model | Sensitivity | Precision | F1-score | FA/24hr |
|-------|-------------|-----------|----------|---------|
| **TKRR-PA** | **0.632 (0.17)** | **0.601 (0.37)** | **0.523 (0.28)** | **5.19 (7.3)** |
| TKRR-PI | 0.630 (0.18) | 0.575 (0.44) | 0.486 (0.33) | 12.0 (23) |
| TKRR-PS | 0.625 (0.14) | 0.414 (0.35) | 0.347 (0.25) | 26.2 (30) |
| SVM-PA | 0.621 (0.15) | 0.555 (0.33) | 0.484 (0.22) | 6.5 (7.1) |
| SVM-PI | 0.620 (0.22) | 0.548 (0.42) | 0.458 (0.31) | 9.06 (14) |
| SVM-PS | 0.623 (0.12) | 0.508 (0.31) | 0.443 (0.21) | 16.6 (22) |

**关键发现**：
- **TKRR-PA在F1-score上最优**（0.523），比TKRR-PI提升7.6%，比TKRR-PS提升50.7%
- **误报率显著降低**：FA/24hr从PI的12.0降至5.19（降低56.7%）
- **与SVM性能相当**：TKRR-PA略优于SVM-PA，但**参数少100倍**，**推理快100倍**

### ROC曲线分析（图3-4）
- **TKRR模型**：PI和PA的AUC均为0.88-0.89，显著高于PS的0.78
- **SVM模型**：PI和PA的AUC为0.88-0.89，PS为0.85
- **结论**：迁移学习有效，且TKRR-PS性能下降更显著（因超参数继承自PI模型）

### 患者级性能（图5）
- **9/15名患者**：PA模型F1-score优于PI和PS
- **5名患者**（33, 65, 78, 82, 99）：PA模型性能下降，这些患者PI模型已表现良好（F1>0.6）且PS模型性能差，说明**过度迁移可能有害**

### 正则化参数μ的影响（表IV, 图6）
| μ值 | F1-score | FA/24hr | 特点 |
|-----|----------|---------|------|
| 1×10^-4 | 0.456 | 11.1 | 接近目标模型，欠拟合 |
| 1×10^-3 | 0.490 | 5.54 | 平衡 |
| **0.01** | **0.523** | **5.19** | **最优折中** |
| 0.1 | 0.512 | 8.04 | 接近源模型 |
| **Optimal** | **0.582** | **4.19** | 按患者调优 |

- **μ=0.01**在合成数据和真实数据上均表现良好，是**鲁棒的默认选择**
- **患者特异性调优**：最优μ值差异大，高μ适合PI模型已好的患者，低μ适合PS模型相对好的患者

### 模型效率（表V）
| Model | 参数数量 | 推理时间 |
|-------|----------|----------|
| SVM-PI | 1.363×10^6 | 3.3×10^-3s |
| **TKRR-PI/PA** | **1.988×10^4** | **3.9×10^-5s** |
| SVM-PS | 1.140×10^4 | 3.3×10^-5s |
| TKRR-PS | 1.988×10^4 | 3.9×10^-5s |

- **参数压缩**：TKRR模型比SVM-PI少**~100倍**参数
- **推理加速**：TKRR比SVM-PI快**~85倍**，适合边缘设备部署

### 消融实验：初始化策略（图7）
- **源模型初始化**：约42次迭代（1轮）即收敛
- **随机初始化**：需~110次迭代收敛，但最终损失略低（差0.011）
- **结论**：源初始化**显著加速收敛**，适合在线适应场景

---

## 4. 关键结论和发现

### 主要发现
1. **有效性**：Adapt-TKM能成功个性化癫痫检测模型，在多数患者上**同时优于PI和PS模型**，F1-score提升7.6%，误报率降低56.7%
2. **高效性**：相比Adaptive SVM，**参数减少100倍，推理速度提升100倍**，适合资源受限的可穿戴设备
3. **隐私保护**：不存储源域数据，避免患者数据共享的隐私风险
4. **鲁棒性**：μ=0.01是跨患者的**鲁棒默认参数**，在合成和真实数据上均表现良好

### 方法局限性
1. **非普适性**：对PI模型已表现良好的患者（F1>0.6），适应可能**降低性能**（5/15患者）
2. **超参数敏感**：μ值选择显著影响性能，**缺乏自动选择策略**；CPD秩R需网格搜索，继承自PI模型可能次优
3. **特征映射复杂性**：需额外选择M（基函数数）和U（边界参数），虽提出启发式方法但仍增加调参负担
4. **数据依赖**：PS模型性能差时，PA模型提升空间有限（如患者13, 61, 83）

### 未来工作方向
1. **自动超参数选择**：开发基于验证集或元学习的μ和R选择策略，实现**无需调优的个性化**
2. **动态秩调整**：研究训练过程中**自适应调整CPD秩**的方法，避免过度参数化
3. **理论分析**：深入理解何时迁移有益/有害，建立**患者适应性预测指标**
4. **在线适应**：探索**设备端持续学习**，利用流式数据实时更新模型
5. **其他张量网络**：尝试Tensor-Train等结构，在**表达能力和效率间更好权衡**
6. **多源迁移**：整合**多个源患者**知识，提升通用性和鲁棒性

### 临床意义
- 为**长期可穿戴癫痫监测**提供了实用方案：低功耗、低延迟、保护隐私
- **少量患者数据**（仅3次发作片段）即可实现有效个性化，降低临床部署门槛
- 事件级评估和误报率优化更符合**临床实际需求**，提升用户接受度

---

### 14. [Beyond Confidence: Adaptive and Coherent Decoding for Diffusion Language Models](https://arxiv.org/abs/2512.02044)

**Authors**: Kecheng Chen, Ziru Liu, Xijia Tao, Hui Liu, Xinyu Fu, Suiyun Zhang, Dandan Tu, Lingpeng Kong, Rui Liu, Haoliang Li  
**Category**: cs.CL  
**Published**: 2025-12-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2512.02044v1  

#### Abstract
Diffusion Language Models (DLMs) have recently achieved significant success due to their any-order generation capabilities. However, existing inference methods typically rely on local, immediate-step metrics such as confidence or entropy which inherently lack a more reliable perspective. This limita...

#### AI Summary (by kimi-k2-thinking)
# Beyond Confidence: Adaptive and Coherent Decoding for Diffusion Language Models - 论文总结

## 1. 主要贡献和创新点

### 解决的问题
- **单步置信度局限性**：现有DLM推理方法依赖局部、即时步骤的置信度或熵指标，缺乏全局视角，容易导致不一致的采样轨迹和次优生成质量
- **理论依据缺失**：当前采样过程缺乏与采样误差率的直接理论联系，难以实现可控的性能保证
- **预算分配次优**：统一解码预算方案效率低下，无法根据生成内容的上下文敏感度动态调整

### 提出的新方法
- **Coherent Contextual Decoding (CCD)**：无需训练的解码框架，通过轨迹修正机制利用历史上下文增强序列一致性
- **自适应采样策略 (CCD-DS)**：基于一致性指标动态调整每步的unmasking预算，替代固定的扩散步骤
- **理论框架**：通过条件互信息（conditional mutual information）建模历史步骤的一致性，建立与采样误差理论上界的联系

### 相比现有方法的优势
- **双重提升**：首次实现推理速度和生成质量的同步提升（最高3.48×加速 + 3.91%性能提升）
- **理论保证**：通过边缘化上下文近似目标分布，提供可计算的采样误差上界控制
- **智能预算分配**：自动识别上下文敏感token（小预算）和上下文不敏感token（大预算）
- **即插即用**：兼容现有DLM和推理优化技术（如remasking、KV-cache加速）

---

## 2. 核心实验方法和设置

### 数据集
- **数学推理**：GSM8K、MATH
- **代码生成**：HumanEval、MBPP
- **规划任务**：Trip planning benchmark（1,600个不同难度示例）

### 实验设置和评估指标
- **基础模型**：LLaDA-8B-Instruct、Dream-7B-Instruct
- **评估指标**：
  - 准确率（Accuracy）
  - 解码步数（Diffusion Steps）
  - 推理效率提升（Speedup）
- **解码配置**：统一预算 $b_t = 1$（每步解码1个token），扩散步数T等于最大生成长度N

### 基线方法对比
- **标准采样**：Dream使用负熵选择token，LLaDA使用最大概率
- **块解码（Block-wise decoding）**：LLaDA在数学基准上采用的半自回归方案
- **零样本设置**：除Trip基准外均采用zero-shot评估

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 模型 | 方法 | GSM8K | MATH | HumanEval | MBPP | Trip Plan |
|------|------|-------|------|-----------|------|-----------|
| **Dream-7B** | Baseline | 81.01 | 40.90 | 52.66 | 58.00 | 15.10 |
| | + CCD | 82.26 (+1.25) | 41.20 (+0.30) | 57.31 (+4.65) | 58.00 | 16.93 (+1.83) |
| | + CCD-DS | 82.51 (+1.50) | 41.20 (+0.30) | 56.71 (+4.05) | 58.00 | **19.01 (+3.91)** |
| **LLaDA-8B** | Baseline | 74.30 | 37.00 | 36.50 | 39.20 | 10.40 |
| | + CCD | 75.30 (+1.00) | 37.20 (+0.20) | 38.41 (+1.91) | 39.20 | 10.80 (+0.40) |
| | + CCD-DS | 75.22 (+0.92) | 37.20 (+0.20) | 38.40 (+1.90) | 39.20 | 11.50 (+1.10) |

### 推理加速效果
- **Dream模型**：
  - MBPP：3.78×加速（270.2步 vs 1,024步）
  - Trip Plan：3.48×加速（75.2步 vs 256步）
  - HumanEval：3.04×加速（253.2步 vs 768步）
- **LLaDA模型**：
  - Trip Plan：2.27×加速（112.5步 vs 256步）
  - GSM8K：1.31×加速（393.0步 vs 512步）

### 消融实验结果
- **Buffer Size影响**：在Trip基准子集上，buffer size=4时达到最优平衡（准确率70%，比baseline提升20.7%，步数减少62.7%）
- **温度系数鲁棒性**：在HumanEval上，温度从0到1.0范围内，CCD-DS均稳定优于baseline，在0.1时达到峰值（56.71% vs 52.66%）
- **动态预算效果**：CCD-DS在保持或提升性能的同时，显著减少解码步数，尤其在生成EOS token的plateau阶段效率提升明显

---

## 4. 关键结论和发现

### 主要发现
1. **历史上下文的价值**：通过滑动窗口历史缓冲存储最近d次迭代的top-V置信token预测，有效近似边缘化目标分布，降低对单步上下文的依赖
2. **一致性即质量**：条件互信息 $I(x_i; \mathbf{c}|\mathbf{s})$ 高的token需要更多上下文确认，应推迟解码；一致性低的token可早期解码
3. **自适应预算的优越性**：CCD-DS能自动识别上下文不敏感区域（如模板、公式化短语）并加大解码预算，对语义模糊token则保守处理
4. **错误轨迹早期拒绝**：在关键分叉点（如数学问题中的"Karen"命名实体），CCD能拒绝单步高置信但语义次优的选择，避免级联错误

### 方法局限性
- **内存开销**：需存储历史预测分布，但通过滑动窗口（d≪T）和top-V筛选（V≪N）将复杂度从 $O(T \times N \times |\mathbb{X}|)$ 降至 $O(d \times V \times |\mathbb{X}|)$
- **超参数依赖**：V和d需针对模型调整（Dream用d=3, V=4；LLaDA用d=2, V=4）
- **早期预测噪声**：扩散初期上下文不足时预测质量较低，但top-V筛选机制可缓解此问题

### 未来工作方向
- **与现有优化技术集成**：探索与remasking、KV-cache加速、speculative decoding等技术的协同效应
- **更大规模验证**：在超过7-8B参数的DLM上测试方法的可扩展性
- **自适应策略深化**：开发更智能的阈值ε选择机制，避免手动调参
- **理论分析扩展**：进一步研究条件互信息与任务难度、模型容量的关系

---

**核心结论**：CCD通过将DLM解码重新定义为基于历史一致性的多步过程，在理论和实践上同时突破了传统单步置信度方法的局限，为DLM推理提供了质量与效率兼得的新范式。

---

### 15. [CREST: Universal Safety Guardrails Through Cluster-Guided Cross-Lingual Transfer](https://arxiv.org/abs/2512.02711)

**Authors**: Lavish Bansal, Naman Mishra  
**Category**: cs.CL  
**Published**: 2025-12-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2512.02711v1  

#### Abstract
Ensuring content safety in large language models (LLMs) is essential for their deployment in real-world applications. However, existing safety guardrails are predominantly tailored for high-resource languages, leaving a significant portion of the world's population underrepresented who communicate i...

#### AI Summary (by kimi-k2-thinking)
# CREST: Universal Safety Guardrails Through Cluster-Guided Cross-Lingual Transfer

## 1. 主要贡献和创新点

### 解决的问题
- **低资源语言安全护栏缺失**：现有安全护栏主要面向高资源语言（如英语、中文、西班牙语等），全球大多数低资源语言缺乏有效保护，导致LLM在这些语言中更容易产生有害内容或被恶意利用
- **计算成本高昂**：现有基于LLM的护栏模型参数过大（通常≥7B），推理速度慢，难以在边缘设备部署
- **数据稀缺**：低资源语言缺乏标注的安全训练数据，传统监督学习方法不可行

### 提出的新方法
- **CREST（CRoss-lingual Efficient Safety Transfer）**：基于聚类指导的跨语言迁移框架
  - 利用**XLM-R**的多语言表示空间对100种语言进行聚类（形成8个语言簇）
  - 仅从每个簇中选择1-2种高资源语言（共13种）作为训练源语言
  - 通过簇内语言的结构相似性实现向低资源语言的零样本迁移

### 相比现有方法的优势
- **参数高效**：CREST-LARGE仅0.5B参数，比现有SOTA护栏小5-16倍
- **语言覆盖广**：支持100种语言，远超PolyGuard（17种）和LlamaGuard3（8种）
- **推理速度快**：比大规模护栏快10倍以上，适合边缘设备部署
- **性能卓越**：在高低资源语言上均达到SOTA水平，零样本迁移能力强

---

## 2. 核心实验方法和设置

### 数据集
**训练数据**：
- **Aegis-AI-Content-Safety-Dataset-2.0**：30k训练样本，覆盖12大类安全风险，翻译为13种训练语言

**评估基准（6个）**：
- **Aegis-CS2**：内容安全测试集
- **HarmBench**：自动化红队测试框架
- **Redteam2k**：2k个红队查询
- **JBB-Behaviors & JBB-Judge**：越狱行为评估
- **StrongReject**：空越狱测试
- **CSRT**：代码切换红队测试数据集

**文化特异性评估**：
- **IndicSafe**：南亚地区文化安全基准
- **Cultural Kaleidoscope**：文化敏感性评估

### 实验设置
- **模型架构**：基于XLM-RoBERTa-Base（279M）和Large（560M）变体，添加单层分类头
- **训练语言（13种）**：西班牙语、英语、德语、俄语、捷克语、芬兰语、印地语、泰米尔语、中文、越南语、阿拉伯语、斯瓦希里语、菲律宾语
- **评估语言**：
  - **In-Domain (ID)**：11种训练语言中的高资源语言
  - **OOD Low-Resource**：11种低资源语言（加利西亚语、冰岛语、南非荷兰语、斯洛文尼亚语、僧伽罗语、泰语、马拉地语、普什图语、爪哇语、豪萨语、格鲁吉亚语）

### 基线方法
| 模型 | 规模 | 语言支持 |
|------|------|----------|
| Aegis-Defensive | 7B | 1（英语） |
| LlamaGuard3 | 8B | 8种语言 |
| PolyGuard-Qwen | 2.5B | 17种语言 |
| DuoGuard-0.5B | 0.5B | 29种语言 |
| PolyGuard-Qwen-Smol | 0.5B | 17种语言 |
| WalledGuard-C | 0.5B | 1（英语） |
| **CREST-BASE** | **0.25B** | **100种语言** |
| **CREST-LARGE** | **0.5B** | **100种语言** |

---

## 3. 主要实验结果和性能指标

### 整体性能对比（F1分数）
**英语基准测试**：
- **CREST-LARGE**在多数基准上**匹配或超越**LlamaGuard3（8B）和Aegis-Defensive（7B）
- 在HarmBench上达到**86.87**，超越所有小规模基线
- 在StrongReject上达到**96.36**，接近SOTA水平

**高资源语言（未在训练中）**：
| 语言 | DuoGuard | LlamaGuard3 | PG-Qwen-Smol | **CREST-LARGE** |
|------|----------|-------------|--------------|-----------------|
| 法语 | 73.81 | 84.07 | - | **86.59** |
| 意大利语 | 61.42 | 83.96 | - | **85.96** |
| 德语 | 76.86 | 82.78 | - | **84.68** |
| 葡萄牙语 | 67.40 | 82.81 | - | **85.28** |
| 西班牙语 | 74.13 | 83.45 | - | **86.55** |

**低资源语言零样本迁移**：
- **CREST-BASE**：ID语言平均F1=82.12，OOD低资源语言=81.73（差距仅0.39）
- **CREST-LARGE**：ID语言平均F1=84.56，OOD低资源语言=84.41（差距仅0.15）
- **关键发现**：零样本迁移性能与有监督的ID语言几乎持平，证明跨簇迁移有效性

**代码切换数据（CSRT）**：
- **CREST-LARGE**以**92.49**的F1分数**超越所有基线**
- 其他小规模基线性能下降显著（DuoGuard仅53.14）

### 消融实验结果

**簇内迁移分析（RQ1）**：
- 在印地语簇内测试：高资源语言（印地语）→ 中资源（卡纳达语）→ 低资源（信德语）
- **印地语训练模型**在15种印度语言上平均F1=**85.62**
- 卡纳达语模型=84.84，信德语模型=78.03
- **结论**：高资源语言作为训练源能实现最优簇内迁移

**跨簇迁移分析（RQ3）**：
- 印地语训练模型在印度簇上表现（86.26）远优于东亚簇（85.07）
- 中文训练模型在东亚簇上表现（85.83）优于印度簇（82.68）
- **印地语+中文联合训练**在两个簇上均达到最佳（87.61和87.36）
- **结论**：同簇迁移 > 跨簇迁移，但多源训练可弥补跨簇差距

**脚本影响分析（RQ2）**：
- **拉丁脚本语言**（加利西亚语、斯洛文尼亚语）表现稳定（平均F1>82）
- **冰岛语**因复杂形态结构导致token碎片化，性能下降（78.52）
- **结论**：脚本重叠和subword tokenization覆盖率显著影响迁移效果

---

## 4. 关键结论和发现

### 主要发现
1. **聚类指导的跨语言迁移是有效的**：通过XLM-R表示空间聚类，仅需13种高资源语言即可实现100种语言的有效覆盖
2. **零样本迁移能力强大**：OOD低资源语言性能与ID语言差距<0.5%，证明结构相似性可弥补数据稀缺
3. **规模与效率的平衡**：0.5B参数的CREST-LARGE在多数任务上超越7B-8B模型，推理速度快10倍
4. **文化适应性存在局限**：在Cultural Kaleidoscope基准上表现（56.79-69.42）低于PG-Qwen（75.71），反映机器翻译可能丢失文化细微差别

### 方法局限性
1. **翻译偏差**：依赖机器翻译可能改变原始有害内容的意图/语气，特别是文化特定表达
2. **推理能力缺失**：作为分类模型，未显式建模复杂推理和上下文理解，可能影响 nuanced safety judgments
3. **形态复杂语言挑战**：对冰岛语等高度屈折语言，因token碎片化导致性能下降
4. **文化特异性不足**：在区域文化安全基准上表现不如大规模模型

### 未来工作方向
1. **上下文化建模**：探索基于轻量级多语言LLM（支持100+语言）的上下文感知安全护栏
2. **文化适应性增强**：开发保留文化细微差别的训练数据收集方法，减少翻译依赖
3. **多模态扩展**：将框架扩展到多模态场景（图像、音频），应对全球用户多样化交互
4. **边缘优化**：进一步压缩模型以适应极端资源受限环境（如IoT设备）

---

**核心价值**：CREST首次在0.5B参数规模下实现100种语言的安全护栏，通过聚类指导的跨语言迁移解决了低资源语言的安全对齐难题，为构建包容性全球AI安全系统提供了可扩展的解决方案。

---

### 16. [SeeNav-Agent: Enhancing Vision-Language Navigation with Visual Prompt and Step-Level Policy Optimization](https://arxiv.org/abs/2512.02631)

**Authors**: Zhengcheng Wang, Zichuan Lin, Yijun Yang, Haobo Fu, Deheng Ye  
**Category**: cs.LG  
**Published**: 2025-12-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2512.02631v1  

#### Abstract
Existing Vision-Language Navigation (VLN) agents based on Large Vision-Language Models (LVLMs) often suffer from perception errors, reasoning errors, and planning errors, which significantly hinder their navigation performance. To address these limitations, a novel VLN agent framework, named SeeNav-...

---

### 17. [Model Recovery at the Edge under Resource Constraints for Physical AI](https://arxiv.org/abs/2512.02283)

**Authors**: Bin Xu, Ayan Banerjee, Sandeep K. S. Gupta  
**Category**: cs.AI  
**Published**: 2025-12-03  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2512.02283v1  

#### Abstract
Model Recovery (MR) enables safe, explainable decision making in mission-critical autonomous systems (MCAS) by learning governing dynamical equations, but its deployment on edge devices is hindered by the iterative nature of neural ordinary differential equations (NODEs), which are inefficient on FP...

---

### 18. [Mirror, Mirror on the Wall -- Which is the Best Model of Them All?](https://arxiv.org/abs/2512.02043)

**Authors**: Dina Sayed, Heiko Schuldt  
**Category**: cs.CL  
**Published**: 2025-12-03  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2512.02043v1  

#### Abstract
Large Language Models (LLMs) have become one of the most transformative tools across many applications, as they have significantly boosted productivity and achieved impressive results in various domains such as finance, healthcare, education, telecommunications, and law, among others. Typically, sta...

---

### 19. [Lightweight Latent Reasoning for Narrative Tasks](https://arxiv.org/abs/2512.02240)

**Authors**: Alexander Gurung, Nikolay Malkin, Mirella Lapata  
**Category**: cs.CL  
**Published**: 2025-12-03  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2512.02240v1  

#### Abstract
Large language models (LLMs) tackle complex tasks by generating long chains of thought or "reasoning traces" that act as latent variables in the generation of an output given a query. A model's ability to generate such traces can be optimized with reinforcement learning (RL) to improve their utility...

---

### 20. [promptolution: A Unified, Modular Framework for Prompt Optimization](https://arxiv.org/abs/2512.02840)

**Authors**: Tom Zehle, Timo Hei{\ss}, Moritz Schlager, Matthias A{\ss}enmacher, Matthias Feurer  
**Category**: cs.CL  
**Published**: 2025-12-03  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2512.02840v1  

#### Abstract
Prompt optimization has become crucial for enhancing the performance of large language models (LLMs) across a broad range of tasks. Although many research papers show its effectiveness, practical adoption is hindered as existing implementations are often tied to unmaintained and isolated research co...

---

### 21. [SpecPV: Improving Self-Speculative Decoding for Long-Context Generation via Partial Verification](https://arxiv.org/abs/2512.02337)

**Authors**: Zhendong Tan, Xingjun Zhang, Chaoyi Hu, Junjie Peng, Kun Xia  
**Category**: cs.LG  
**Published**: 2025-12-03  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2512.02337v1  

#### Abstract
Growing demands from tasks like code generation, deep reasoning, and long-document understanding have made long-context generation a crucial capability for large language models (LLMs). Speculative decoding is one of the most direct and effective approaches for accelerating generation. It follows a ...

---

### 22. [GoRL: An Algorithm-Agnostic Framework for Online Reinforcement Learning with Generative Policies](https://arxiv.org/abs/2512.02581)

**Authors**: Chubin Zhang, Zhenglin Wan, Feng Chen, Xingrui Yu, Ivor Tsang, Bo An  
**Category**: cs.LG  
**Published**: 2025-12-03  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2512.02581v1  

#### Abstract
Reinforcement learning (RL) faces a persistent tension: policies that are stable to optimize are often too simple to represent the multimodal action distributions needed for complex control. Gaussian policies provide tractable likelihoods and smooth gradients, but their unimodal form limits expressi...

---

### 23. [Assessing the performance of correlation-based multi-fidelity neural emulators](https://arxiv.org/abs/2512.02868)

**Authors**: Cristian J. Villatoro, Gianluca Geraci, Daniele E. Schiavazzi  
**Category**: cs.LG  
**Published**: 2025-12-03  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2512.02868v1  

#### Abstract
Outer loop tasks such as optimization, uncertainty quantification or inference can easily become intractable when the underlying high-fidelity model is computationally expensive. Similarly, data-driven architectures typically require large datasets to perform predictive tasks with sufficient accurac...

---

### 24. [The 4/$\delta$ Bound: Designing Predictable LLM-Verifier Systems for Formal Method Guarantee](https://arxiv.org/abs/2512.02080)

**Authors**: PIerre Dantas, Lucas Cordeiro, Youcheng Sun, Waldir Junior  
**Category**: cs.AI  
**Published**: 2025-12-03  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2512.02080v1  

#### Abstract
The idea of using Formal Verification tools with large language models (LLMs) has enabled scaling software verification beyond manual workflows. However, current methods remain unreliable. Without a solid theoretical footing, the refinement process can wander; sometimes it settles, sometimes it loop...

---

### 25. [DialogGuard: Multi-Agent Psychosocial Safety Evaluation of Sensitive LLM Responses](https://arxiv.org/abs/2512.02282)

**Authors**: Han Luo, Guy Laban  
**Category**: cs.AI  
**Published**: 2025-12-03  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2512.02282v1  

#### Abstract
Large language models (LLMs) now mediate many web-based mental- health, crisis, and other emotionally sensitive services, yet their psychosocial safety in these settings remains poorly understood and weakly evaluated. We present DialogGuard, a multi-agent frame- work for assessing psychosocial risks...

---

### 26. [From Moderation to Mediation: Can LLMs Serve as Mediators in Online Flame Wars?](https://arxiv.org/abs/2512.03005)

**Authors**: Dawei Li, Abdullah Alnaibari, Arslan Bisharat, Manny Sandoval, Deborah Hall, Yasin Silva, Huan Liu  
**Category**: cs.AI  
**Published**: 2025-12-03  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2512.03005v1  

#### Abstract
The rapid advancement of large language models (LLMs) has opened new possibilities for AI for good applications. As LLMs increasingly mediate online communication, their potential to foster empathy and constructive dialogue becomes an important frontier for responsible AI research. This work explore...

---

### 27. [DOLMA: A Data Object Level Memory Disaggregation Framework for HPC Applications](https://arxiv.org/abs/2512.02300)

**Authors**: Haoyu Zheng, Shouwei Gao, Jie Ren, Wenqian Dong  
**Category**: cs.DC  
**Published**: 2025-12-03  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2512.02300v1  

#### Abstract
Memory disaggregation is promising to scale memory capacity and improves utilization in HPC systems. However, the performance overhead of accessing remote memory poses a significant chal- lenge, particularly for compute-intensive HPC applications where execution times are highly sensitive to data lo...

---

### 28. [InstructLR: A Scalable Approach to Create Instruction Dataset for Under-Resourced Languages](https://arxiv.org/abs/2512.02213)

**Authors**: Mamadou K. Keita, Sebastien Diarra, Christopher Homan, Seydou Diallo  
**Category**: cs.LG  
**Published**: 2025-12-03  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2512.02213v1  

#### Abstract
Effective text generation and chat interfaces for low-resource languages (LRLs) remain a challenge for state-of-the-art large language models (LLMs) to support. This is mainly due to the difficulty of curating high-quality instruction datasets for LRLs, a limitation prevalent in the languages spoken...

---

### 29. [Pruning AMR: Efficient Visualization of Implicit Neural Representations via Weight Matrix Analysis](https://arxiv.org/abs/2512.02967)

**Authors**: Jennifer Zvonek, Andrew Gillette  
**Category**: cs.LG  
**Published**: 2025-12-03  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2512.02967v1  

#### Abstract
An implicit neural representation (INR) is a neural network that approximates a spatiotemporal function. Many memory-intensive visualization tasks, including modern 4D CT scanning methods, represent data natively as INRs. While INRs are prized for being more memory-efficient than traditional data st...

---

### 30. [ProteinPNet: Prototypical Part Networks for Concept Learning in Spatial Proteomics](https://arxiv.org/abs/2512.02983)

**Authors**: Louis McConnell, Jieran Sun, Theo Maffei, Raphael Gottardo, Marianna Rapsomaniki  
**Category**: cs.LG  
**Published**: 2025-12-03  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2512.02983v1  

#### Abstract
Understanding the spatial architecture of the tumor microenvironment (TME) is critical to advance precision oncology. We present ProteinPNet, a novel framework based on prototypical part networks that discovers TME motifs from spatial proteomics data. Unlike traditional post-hoc explanability models...

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

# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2025-12-03 07:15:42 UTC
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
# Fantasy: Efficient Large-scale Vector Search on GPU Clusters with GPUDirect Async 论文总结

## 1. 主要贡献和创新点

### 解决的问题
- **单GPU内存容量限制**：十亿级向量索引（如SIFT1B）需要数百GB内存，远超单GPU HBM容量（通常40-80GB）
- **CPU-GPU架构瓶颈**：现有out-of-core方法将向量存储在CPU内存或SSD中，数据加载过程会阻塞GPU计算，导致吞吐量受限
- **通信效率低下**：传统GPU集群搜索需要CPU参与数据传输控制路径，引入额外延迟和开销

### 提出的新方法
**Fantasy系统**：基于GPU集群的分布式向量搜索架构，核心创新包括：
- **GPUDirect Async (IBGDA)通信**：利用InfiniBand GPUDirect Async实现GPU到GPU的RDMA直接通信，绕过CPU和GPU SM，达到线速传输（400Gb/s）
- **两阶段微批处理流水线**：将查询批次分为两个microbatch，实现K-means路由、向量分发、并行搜索和结果合并四个阶段的流水线重叠执行
- **K-means空间分区**：使用K-means聚类对向量空间进行分区，指导查询路由到最相关的GPU节点（top-c集群）

### 相比现有方法的优势
- **计算通信重叠**：通过IBGDA异步传输，GPU SMs在通信期间保持空闲，可执行其他计算任务，显著提升端到端吞吐量
- **支持大批量查询**：In-HBM架构使HBM带宽成为首要瓶颈（而非PCIe/NVMe），可支撑更大batch size（bs），提高平均吞吐量
- **消除数据移动**：图结构和向量完全驻留HBM，查询执行无需额外数据传输
- **高可扩展性**：分布式设计支持千亿级向量索引跨多GPU节点存储和搜索

---

## 2. 核心实验方法和设置

**注**：提供的PDF内容在方法论部分（第5页）结束，**未包含完整实验结果章节**。以下基于文中理论分析和配置描述：

### 数据集
- 文中提及**SIFT1B**（十亿规模）作为典型大规模数据集示例
- 向量维度d=1536（符合现代LLM embedding维度）
- 使用FP32/FP16精度格式

### 实验设置
- **硬件配置**：NVIDIA A100 GPU（HBM带宽1.55TB/s，峰值算力156TFLOP/s）
- **集群拓扑**：16 ranks跨2节点，每节点8 GPUs通过NVLink（600GB/s）互联，节点间通过IBGDA RDMA（25GB/s/GPU）
- **评估指标**：查询吞吐量（QPS）、延迟（ms）、召回率（recall）、算术强度（AI）

### 基线方法对比
- **Out-of-core搜索**：图结构存CPU内存/SSD，按需加载到GPU
- **单GPU CAGRA**：cuVS库中的GPU加速图搜索算法
- **In-HBM集体搜索**：纯GPU内存方法（Fantasy的对比基准）

---

## 3. 主要实验结果和性能指标（理论分析）

由于原文缺失实验结果图表，以下基于**第2-3章的理论建模和估算**：

### 各阶段延迟估算（batch size bs=10k, d=1536, c=3）

| 阶段 | 计算/通信量 | 估算时间 | 瓶颈分析 |
|------|-------------|----------|----------|
| **Stage 1: K-means路由** | 2×10k×1536×4096 ≈ 2.52×10¹¹ FLOPs | **1.35 ms** | Compute-bound（TF32 Tensor Cores） |
| **Stage 2: All-to-All分发** | 97.3MB intra-node + 86.6MB inter-node | **3.67 ms** | **RDMA带宽主导**（NVLink贡献可忽略） |
| **Stage 3: 并行搜索** | 每query访问1152个邻居，3.539MB/query | **68.5 ms** | **HBM带宽主导**（内存密集型） |
| **Stage 4: 结果合并** | 约为分发阶段的c倍（c=3） | **11.01 ms** | 网络带宽主导 |

### 关键性能数据
- **单rank搜索吞吐量**：**4.37×10⁵ queries/s**（受HBM带宽限制）
- **算术强度**：AI ≈ **0.5-0.75 FLOP/byte**（FP32），证实为**内存密集型**任务
- **端到端延迟**：约**84.5 ms**（1.35 + 3.67 + 68.5 + 11.01）处理30k向量（c×bs）
- **通信占比**：分发+合并约**14.68 ms**，仅占总延迟**17.4%**，验证计算通信重叠的有效性

### 架构对比结论
- **Out-of-core**：性能受限于PCIe/NVMe带宽（~32-64GB/s），batch size增大时性能下降
- **In-HBM集体搜索**：性能受限于HBM带宽（~1.5TB/s），可支撑**数量级更大的batch size**，吞吐量显著提升

---

## 4. 关键结论和发现

### 主要发现
1. **内存带宽是核心瓶颈**：图搜索的算术强度极低（<1 FLOP/byte），优化重点应放在最大化内存带宽利用率而非计算单元
2. **IBGDA通信开销可隐藏**：通过两阶段流水线，网络传输延迟（~3.67ms）可被计算延迟（~68.5ms）重叠，几乎不成为系统瓶颈
3. **K-means路由效率极高**：子毫秒级延迟（1.35ms）使查询分发开销可忽略不计
4. **分布式HBM扩展性优越**：将索引分片存储在多GPU HBM中，避免了慢速存储介质，吞吐量随节点数线性扩展

### 方法局限性
- **硬件依赖性强**：需要支持IBGDA的InfiniBand网卡和GPU（如Hopper架构），部署成本高
- **K-means分区质量**：查询路由精度依赖K-means聚类效果，对非均匀分布数据可能产生负载不均衡
- **静态分区**：图结构分区后固定，难以动态适应数据分布变化或在线更新
- **通信同步开销**：All-to-All通信在极端规模下仍可能引入同步延迟（文中假设理想无拥塞）

### 未来工作方向
- **动态负载均衡**：研究运行时自适应分区调整机制，应对查询热点和倾斜分布
- **量化压缩**：探索INT8/INT4量化进一步降低HBM带宽需求和通信量
- **多租户隔离**：在共享GPU集群上实现查询隔离和资源保证
- **端到端集成**：与RAG系统深度耦合，优化检索与生成的协同调度

---

**总结**：Fantasy通过**IBGDA硬件加速通信**和**微批处理流水线**，解决了大规模向量搜索的内存墙问题，在理论上可实现**高吞吐量、低延迟**的十亿级向量检索。尽管缺乏完整实验数据验证，其架构设计和理论分析为分布式GPU向量搜索提供了重要参考。

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
# 论文总结：《From monoliths to modules: Decomposing transducers for efficient world modelling》

## 1. 主要贡献和创新点

### 解决的问题
论文针对**世界模型（world models）**在复杂环境中面临的高计算成本、低可解释性和难以分布式推理的问题，提出将单体（monolithic）世界模型分解为模块化组件的框架。核心挑战在于：如何在保持预测能力的同时，将纠缠的高维动态系统分解为功能独立的子系统。

### 提出的新方法
1. **Transducer分解框架**：基于计算力学（computational mechanics）理论，将环境建模为**transducer**（一种推广POMDP的输入-输出模型），并首次系统研究了其逆向分解过程。
2. **信息论诊断工具**：
   - **Intransducibility**：当潜变量可访问时，量化一个过程无法被transducer生成的程度
   - **Acausality**：仅基于可观测数据，测量接口违反非预期性（non-anticipatory）条件的程度
3. **分解算法**：提出两种递归分解算法（Algorithm 1和2），通过"剥离"模块的方式构建有向无环图（DAG）表示的因果架构。
4. **粗粒化（Coarse-graining）理论**：证明在保持边际接口的前提下，如何对transducer网络进行状态空间约简，实现多尺度建模。

### 相比现有方法的优势
- **结构性透明**：相比黑盒神经网络，分解后的模块对应明确的功能子系统，提升可解释性
- **计算效率**：支持**并行化推理**，各子transducer可独立更新信念状态，避免全局高维潜变量推理
- **理论保证**：证明ε-transducer（最小预测表示）在组合下保持最小性，为模块化提供理论基础
- **无需预设结构**：通过数据驱动的信息论度量自动发现因果依赖，而非依赖人工设计的归纳偏置

---

## 2. 核心实验方法和设置

**重要说明**：这是一篇**纯理论论文**，未包含传统意义上的实验验证。论文聚焦于数学框架和算法设计，而非实证评估。

### 理论验证方式
- **数学证明**：通过定理和命题形式化分解条件（Theorem 1-2, Lemma 1-2）
- **算法描述**：提供伪代码（Algorithm 1-2）说明分解流程
- **示例说明**：使用合成示例（如Figure 4, 6, 9）图解概念，但未在真实数据集上测试

### 评估指标（理论层面）
- **Intransducibility值**：判断分解是否精确（值为0表示可分解）
- **Acausality值**：判断接口是否因果一致（值为0表示有效transducer）
- **模块稀疏性**：分解后网络的连接密度

---

## 3. 主要实验结果和性能指标

**无实验数据**。论文未报告任何数值结果、性能对比或消融研究。所有结论均基于理论推导。

---

## 4. 关键结论和发现

### 主要理论发现
1. **分解可行性**：任何满足非预期性条件的因果接口都存在transducer表示，且当且仅当Intransducibility/Acausality为零时可被精确分解。
2. **层次化结构**：复杂世界模型可递归分解为**prime transducers**（不可再分的原子模块），形成层次化的DAG结构。
3. **最小表示的封闭性**：ε-transducer（最优预测模型）在组合下保持最小性，即组合后的ε-transducer仍是复合接口的最小预测器。
4. **粗粒化条件**：仅当节点块在因果上相邻（所有外部节点全在上游或下游）时，才能无损地聚合或消除节点。

### 方法局限性
1. **计算不可行性**：Intransducibility和Acausality需要估计长历史序列的联合分布，实际中需有限时间窗或变分近似。
2. **假设限制**：当前框架仅适用于**前馈（feedforward）**和**机制平稳（mechanistically stationary）**的transducer，无法处理反馈、非平稳性和自适应系统。
3. **分解非唯一性**：可能存在多个最小分解方案，算法结果依赖于打破平局（tie-breaking）的策略。
4. **缺乏实证验证**：未在真实世界模型（如RL环境、Transformer）上验证有效性。

### 未来工作方向
1. **扩展框架**：处理反馈循环、时变动力学和自适应agent
2. **近似分解**：开发有限样本下的统计检验，区分真实结构与采样噪声
3. **实证应用**：在训练好的世界模型、RL环境和大规模神经网络中验证理论
4. **算法优化**：设计可扩展的估计算法，降低计算复杂度
5. **与神经科学联系**：探索生物智能是否采用类似的模块化transducer架构

---

## 总结

该论文为世界模型的模块化提供了**严谨的理论基础**，其核心贡献是信息论分解框架而非实验突破。虽然缺乏实证结果，但为AI安全、可解释性和高效推理指明了新方向，期待未来工作能在实际系统中验证这些理论工具的有效性。

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
- **开源模型性能差距**：开源LLM与闭源前沿模型（如GPT-5、Gemini-3.0-Pro）在复杂任务上的性能差距持续扩大
- **长上下文效率瓶颈**：传统vanilla attention机制在长序列上的计算复杂度严重制约部署和后训练效率
- **Agent能力滞后**：开源模型在工具使用场景下的泛化和指令遵循能力显著落后于专有模型
- **后训练计算不足**：开源社区在后训练阶段投入的计算资源有限，制约了模型在困难任务上的表现

### 提出的新方法
- **DeepSeek Sparse Attention (DSA)**：一种高效的稀疏注意力机制，通过lightning indexer和细粒度token选择机制，将核心注意力复杂度从$O(L^2)$降至$O(Lk)$（$k \ll L$）
- **可扩展RL框架**：基于GRPO算法，实现稳定的大规模强化学习训练，后训练计算预算超过预训练成本的10%
- **大规模Agent任务合成流水线**：自动生成1,827个任务环境和85,000个复杂提示，支持可扩展的Agent后训练
- **Thinking Context Management**：专为工具调用场景设计的上下文管理策略，在保留推理内容的同时避免token冗余

### 相比现有方法的优势
- **计算效率**：DSA在长上下文场景下实现显著端到端加速，同时保持与密集注意力相当的性能
- **性能突破**：DeepSeek-V3.2-Speciale在IMO 2025、IOI 2025、ICPC WF 2025等顶级竞赛中达到金牌水平
- **成本效益**：在Agent场景中提供高性价比的替代方案，性能接近前沿闭源模型但成本显著降低
- **泛化能力**：合成任务驱动的RL训练使模型能有效泛化到未见过的Agent场景

## 2. 核心实验方法和设置

### 评估数据集
**推理能力基准**：
- MMLU-Pro、GPQA Diamond、Human Last Exam (HLE)
- AIME 2025、HMMT Feb/Nov 2025、IMOAnswerBench
- LiveCodeBench、Codeforces

**Agent能力基准**：
- Terminal Bench 2.0、SWE-Verified、SWE-Multilingual
- BrowseComp/BrowseCompZh、τ2-bench
- MCP-Universe、MCP-Mark、Tool-Decathlon

**竞赛评估**：
- IMO 2025、CMO 2025、IOI 2025、ICPC WF 2025

### 实验设置
- **模型变体**：DeepSeek-V3.2（标准版）、DeepSeek-V3.2-Speciale（高计算版）
- **上下文长度**：128K tokens
- **采样温度**：1.0
- **评估模式**：thinking mode（带推理链）和non-thinking mode（直接回答）
- **工具格式**：标准function calling格式

### 基线对比
**闭源模型**：GPT-5-High、Claude-4.5-Sonnet、Gemini-3.0-Pro  
**开源模型**：Kimi-K2-Thinking、MiniMax-M2

## 3. 主要实验结果和性能指标

### 关键性能数据

| 基准测试 | DeepSeek-V3.2 | GPT-5-High | Gemini-3.0-Pro | 最佳开源基线 |
|---------|---------------|------------|----------------|--------------|
| **AIME 2025** | 93.1% | 94.6% | 95.0% | 94.5% (Kimi-K2) |
| **HMMT Feb 2025** | 92.5% | 88.3% | 97.5% | 89.4% (Kimi-K2) |
| **LiveCodeBench** | 83.3% | 84.5% | 90.7% | 83.0% (MiniMax) |
| **Codeforces** | 2,386 Rating | 2,537 | 2,708 | - |
| **Terminal Bench 2.0** | 46.4% | 35.2% | 54.2% | 35.7% (Kimi-K2) |
| **SWE-Verified** | 73.1% | 74.9% | 76.2% | 71.3% (Kimi-K2) |
| **BrowseComp** | 67.6%* | 54.9% | - | 60.2%* (Kimi-K2) |

*使用上下文管理技术

### DeepSeek-V3.2-Speciale突破性表现
- **竞赛成绩**：
  - IMO 2025：35/42分（金牌阈值）
  - IOI 2025：492/600分（金牌，排名第10）
  - ICPC WF 2025：10/12题（金牌，排名第2）
  - CMO 2025：102/126分（金牌）

- **基准测试**：在AIME (96.0%)、HMMT Feb (99.2%)上超越Gemini-3.0-Pro

### 效率对比
- **推理成本**：长上下文场景下，DSA实现显著端到端加速（图3）
  - Prefilling阶段：128K上下文时成本降低约70%
  - Decoding阶段：成本降低约60%
- **Token效率**：DeepSeek-V3.2比Kimi-K2-Thinking少用约30-40% tokens达到相当性能

### 消融实验结果
**合成任务有效性**：
- 合成任务对DeepSeek-V3.2-Exp仅12%准确率，对前沿闭源模型最高62%（表5）
- 仅在合成任务上做RL，在τ2-bench、MCP-Mark等真实基准上提升显著（图5）
- 相比仅在code/search环境训练，合成任务带来更好的泛化能力

**上下文管理策略**：
- BrowseComp上，Discard-all策略达到67.6%准确率，接近并行扩展但token使用更少
- Summary策略平均扩展至364步，性能达60.2%

**Thinking模式效果**：
- Thinking模式在Agent任务上普遍优于non-thinking模式（表9）
  - Terminal Bench：46.4% vs 37.1%
  - MCP-Mark：38.0% vs 26.5%

## 4. 关键结论和发现

### 主要发现
1. **稀疏注意力可行**：DSA在保持与密集注意力性能相当的同时，显著降低了长上下文计算成本，解决了效率瓶颈
2. **RL计算规模定律**：后训练计算预算超过预训练成本10%时，推理能力持续提升，验证了RL计算规模化的有效性
3. **合成数据价值**：大规模合成Agent任务能有效提升模型在真实环境中的泛化能力，克服数据稀缺问题
4. **推理与工具使用融合**：通过冷启动和上下文管理，thinking模式可无缝集成到工具调用场景，提升复杂任务解决能力
5. **开源模型突破**：DeepSeek-V3.2-Speciale首次在IMO、IOI等顶级竞赛中达到金牌水平，标志着开源模型进入新阶段

### 方法局限性
1. **知识广度不足**：由于总训练FLOPs较少，世界知识储备仍落后于Gemini-3.0-Pro等前沿模型
2. **Token效率待提升**：生成轨迹通常比Gemini-3.0-Pro更长，需要优化推理链的"智能密度"
3. **复杂任务解决能力**：在最具挑战性的任务上仍略逊于前沿闭源模型
4. **上下文管理兼容性**：当前thinking模式与Terminus等Agent框架存在兼容性问题

### 未来工作方向
1. **扩大预训练规模**：通过增加预训练计算弥补知识广度差距
2. **优化Token效率**：研究更高效的推理链生成策略，减少冗余自验证
3. **混合扩展策略**：探索串行（上下文管理）与并行扩展的最佳组合，最大化效率和可扩展性
4. **Agent框架适配**：改进上下文管理以兼容更多Agent框架
5. **推理能力深化**：进一步增加RL计算预算，探索推理能力的极限

---

**核心结论**：DeepSeek-V3.2通过DSA架构创新、规模化RL训练和大规模Agent任务合成，在保持计算效率的同时，将开源模型的推理和Agent能力推向新高度，其Speciale变体在顶级竞赛中达到金牌水平，显著缩小了开源与闭源模型的差距。

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

#### AI Summary (by kimi-k2-thinking)
# Fast-Decoding Diffusion Language Models via Progress-Aware Confidence Schedules 论文总结

## 1. 主要贡献和创新点

### 解决的问题
扩散大语言模型（dLLMs）虽然提供了并行生成、灵活填充和双向注意力等优势，但其**迭代式采样过程计算开销巨大**，严重限制了实际应用。现有方法要么需要额外训练，要么依赖复杂启发式规则，且在不同任务上表现不稳定，特别是在长文本生成中容易过早终止导致质量下降。

### 提出的新方法：SchED
论文提出**SchED（Schedule-based Early Decoding）**，一种**训练无关、模型无关**的早期退出算法：
- **核心思想**：将扩散解码视为"何时停止"问题，而非固定步数执行
- **技术实现**：聚合答案区域的全序列logit margin（top-2 logits差值），并与**平滑的、进度依赖的置信度阈值**比较
- **调度函数**：提供linear、cosine、exponential三种单调非增阈值调度，从初始高阈值τ_high平滑放松到最终低阈值τ_low

### 相比现有方法的优势
1. **无需训练**：直接作用于预训练dLLM，不改变模型参数
2. **架构无关**：适用于单块解码器（Dream）和块扩散（LLaDA）等不同dLLM架构
3. **全序列稳定**：聚合整个答案区域而非局部前缀，避免局部置信度尖峰导致的过早退出
4. **进度感知**：阈值随归一化扩散进度p=t/T动态调整，适应不同任务的收敛特性
5. **鲁棒性强**：在长文本生成任务上显著优于Prophet等基线方法

---

## 2. 核心实验方法和设置

### 评估模型
- **Dream家族**：单块解码器，评估Base和Instruct两个变体（7B规模）
- **LLaDA家族**：块扩散架构，评估Base和Instruct两个变体

### 基准数据集（10个）
覆盖多类下游任务：
- **多选题**：MMLU、HellaSwag、PIQA、Winogrande、GPQA
- **数学推理**：GSM8K
- **长文本生成**：LongBench-HotpotQA（F1）、LongBench-MultiNews（ROUGE）
- **机器翻译**：WMT14 En-Fr、WMT16 En-De（CHRF）

### 实验配置
- **最大步数**：MCQ任务T=5，数学/翻译T=256，长文本T=512
- **生成长度**：根据任务设置5-512 tokens
- **阈值设置**：固定τ_high=7.5，评估τ_low∈{0, 2.5}两种模式
- **调度类型**：linear、cosine、exponential（k∈{2,4,8,16}）

### 评估指标
- **质量指标**：各任务标准指标（准确率/F1/ROUGE/CHRF）
- **效率指标**：相对于基线的**加速比（speedup）**
- **综合指标**：**QPS（Quality-Penalized Speed）**
  ```
  QPS_γ = speedup × (score / baseline_score)^γ
  ```
  采用γ=4保守惩罚质量损失，强调高保真度

### 基线对比
1. **Standard Diffusion**：固定步数T的完整解码
2. **Prophet**：基于top-2 logit gap的离散早期退出规则

---

## 3. 主要实验结果和性能指标

### Dream Instruct模型结果（表2）
**保守设置（τ_low=2.5）**：
- **平均加速**：3.8-4.0×（linear/cosine/exp-k=2）
- **质量保持**：99.8-100%基线性能（平均得分58.16 vs 58.20）
- **最佳QPS**：**4.30**（exp-k=16, τ_low=2.5），加速4.48×且得分57.59（仅降1%）

**激进设置（τ_low=0）**：
- **最高加速**：exp-k=16达**18.6×**（GPQA），但HellaSwag/PIQA质量显著下降
- **翻译任务**：保持CHRF接近基线，加速2.3-2.8×

### Dream Base模型结果（表1）
**保守设置**：
- **稳定加速**：1.04-1.14×，质量几乎无损（55.31-55.32 vs 55.31）
- **任务提升**：exp-k=2在WMT14 En-Fr上提升+0.27分

**激进设置**：
- **最高加速**：exp-k=16达**2.34×**，但质量下降约2%（53.36 vs 55.31）
- **长文本优势**：HotpotQA加速2.78×且F1提升+7.1

### QPS对比（表3）
| 方法 | Dream Base QPS | Dream Instruct QPS |
|------|----------------|-------------------|
| **SchED (exp-k=16, τ_low=2.5)** | 1.07 | **4.30** |
| **SchED (cosine, τ_low=0)** | 1.12 | 3.83 |
| **Prophet** | 1.07 | 2.91 |

**结论**：SchED在所有配置下QPS均显著优于Prophet，尤其在Instruct模型上优势更明显。

### LLaDA模型结果（附录表7-8）
- **Instruct变体**：exp-k=4(7.5,2.5)达2.13×加速，得分53.17（接近基线52.28）
- **Base变体**：加速更激进（最高10.87×），但质量损失随k值增大而增加

---

## 4. 关键结论和发现

### 主要发现
1. **指令微调的关键作用**：Instruct模型通过**加速预测熵衰减**实现更大加速空间。熵分析显示（图1），Instruct模型在QA任务上初始熵更高但下降更快，而Base模型熵轨迹更平坦、重叠度更高

2. **进度感知调度的有效性**：平滑的阈值放松策略比离散规则更稳定，避免硬相位切换的脆弱性。Cosine/linear调度在激进与保守间取得良好平衡

3. **全序列聚合的重要性**：相比Prophet的局部置信度计算，SchED的**全答案区域聚合**在长文本生成中更鲁棒，防止因早期局部稳定而提前终止

4. **任务依赖性**：
   - **MCQ任务**：因选项明确，置信度建立快，适合激进调度
   - **数学推理**（GSM8K）：需逐步构建解，需要更多步数
   - **长文本生成**：最受益于平滑调度和全序列聚合
   - **翻译**：双向精炼使其在保持CHRF的同时有效加速

### 方法局限性
1. **超参数敏感性**：τ_high、τ_low、k需针对任务调整，虽提供"旋钮"但增加调参成本
2. **质量-效率权衡**：激进调度（如exp-k=16, τ_low=0）虽加速显著，但会在部分任务（HellaSwag、PIQA）上造成明显质量损失
3. **推理时局限**：SchED仅作用于推理阶段，未探索与模型联合训练或学习最优调度
4. **未结合其他加速技术**：未与speculative decoding、dKV-Cache等正交优化集成

### 未来工作方向
1. **学习调度参数**：通过元学习或强化学习自动优化τ(p)函数
2. **任务自适应聚合**：设计基于任务结构的动态聚合策略（如数学推理中关注关键步骤token）
3. **领域感知阈值**：为不同领域（代码、数学、创意写作）定制专用调度
4. **混合加速**：结合推测解码、缓存机制实现进一步加速
5. **理论分析**：深入研究扩散过程中置信度与生成质量的数学关系

---

**核心贡献总结**：SchED通过**进度感知的平滑置信度调度**，将dLLMs的迭代解码转化为智能停止问题，在**指令微调模型上实现4倍加速且几乎无损性能**，为dLLMs的高效部署提供了简单鲁棒的解决方案。

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
# 论文核心结论与实验结果总结

## 1. 主要贡献和创新点

### 解决的问题
- **Boltzmann Machines (BMs)训练计算成本高**：传统MCMC采样因顺序更新规则难以并行化，严重限制了更具表达力的BM（如Semi-Restricted BMs, SRBMs）的实际应用
- **有效逆温度βeff估计困难**：基于优化机的采样器（如量子退火器、SB算法）无法直接控制输出分布的逆温度，而传统KL散度最小化方法计算复杂度指数级增长，不适用于学习循环

### 提出的新方法
1. **Langevin Simulated Bifurcation (LSB)**：一种受量子启发组合优化算法simulated bifurcation启发的并行化采样器
   - 通过离散化和随机初始化改进传统SB算法
   - 保持与MCMC相当的采样精度，同时实现全变量并行更新

2. **Conditional Expectation Matching (CEM)**：高效的βeff估计方法
   - 利用SRBM中隐藏变量在可见变量固定时的条件独立性
   - 通过匹配采样得到的条件期望与解析表达式来估计βeff

3. **Sampler-Adaptive Learning (SAL)**：结合LSB和CEM的统一学习框架
   - 动态适应采样器输出的有效逆温度
   - 支持FBM、RBM和SRBM的高效训练

### 相比现有方法的优势
- **并行性**：LSB支持所有变量同时更新，计算速度显著优于顺序MCMC
- **准确性**：LSB采样精度与Gibbs采样相当（甚至略优），DKL平均0.07±0.01 vs 0.09±0.02
- **可扩展性**：CEM估计βeff的相对误差仅~3.6%，且计算成本远低于KL方法
- **表达力**：首次实现SRBM的高效训练，突破RBM的表达能力限制

---

## 2. 核心实验方法和设置

### 数据集
| 数据集 | 类型 | 规模 | 特点 |
|--------|------|------|------|
| **3-spin模型** | 合成数据 | Nv=10, D=9600 | 随机场玻璃系统，能量景观复杂，可精确计算DKL |
| **Bars-and-Stripes (BAS)** | 合成图像 | 7×6=42像素, D=96 | 192种条纹模式，测试生成与重建能力 |
| **OptDigits** | 真实手写数字 | 8×8=64像素+10类标签, D=3823 | 74维Ising向量，更具挑战性的真实任务 |

### 实验设置
- **模型结构**：FBM (Nh=0), RBM (Nh=5或21), SRBM (Nh=5或21)
- **采样参数**：LSB迭代次数M=100，Δ=1，σ通过候选集{0.5,...,2.0}优化
- **训练参数**：学习率η∈{0.05,0.001}，动量α=0.5，L2正则化λ=10⁻⁵
- **评估指标**：KL散度DKL(PD||Qβ)、生成样本质量、重建错误率、分类准确率

### 基线方法对比
- **CD-100**：标准RBM训练方法，100步blocked Gibbs采样
- **Gibbs采样**：作为LSB精度对比的黄金标准
- **其他SB变体**：aSB, bSB, dSB, cLSB（LSB的clip变体）
- **MLPL**：最大对数伪似然估计（替代βeff估计方法）

---

## 3. 主要实验结果和性能指标

### 3.1 采样性能评估（3-spin模型）
- **LSB vs Gibbs**：在10个随机实例中，LSB在6/10实例上更优，平均DKL为**0.07±0.01**，Gibbs为**0.09±0.02**
- **βeff估计**：CEM与KL方法的平均有符号相对误差仅**3.6%**，且无系统偏差

### 3.2 学习性能对比（3-spin模型）
- **SRBM优势**：SRBM(SAL)的DKL显著低于RBM(CD-100)和FBM(SAL)
- **收敛速度**：SRBM在~600 epoch达到稳定，最终成本函数比RBM低约15-20%
- **消融实验**：
  - 用MLPL替代CEM：SRBM性能降至RBM水平
  - CD-1/CD-10训练SRBM：严重不稳定，DKL急剧上升
  - DMFI改进CD：性能仍低于RBM(CD-1)

### 3.3 生成与重建（BAS数据集）
- **生成质量**：6000 epoch后生成样本**100%为有效BAS模式**，无错误生成
- **重建性能**：1000 epoch后缺失像素(47.6%)的错误率降至**0.5%**，接近完美重建
- **过拟合现象**：3000 epoch后错误率轻微回升至~2%

### 3.4 真实数据表现（OptDigits）
- **无条件生成**：500 epoch后生成样本已具备清晰数字结构
- **条件生成**：3000 epoch后类条件生成样本**视觉可识别率>90%**
- **分类准确率**：通过条件采样推断标签，**最终准确率≈90%**（从随机10%基线快速提升）

---

## 4. 关键结论和发现

### 主要发现
1. **SRBM的实用化**：SAL首次使SRBM训练在计算效率和模型性能上均超越RBM，验证了增加visible-visible连接的价值
2. **LSB的有效性**：在复杂自旋玻璃系统上，并行LSB采样精度可与顺序Gibbs采样媲美，甚至略优
3. **CEM的可靠性**：CEM提供稳定、准确的βeff估计，且计算开销可忽略，是SAL框架的关键使能技术
4. **泛化能力**：SAL训练的SRBM展现出强泛化能力，在BAS上实现近乎完美的分布学习而非样本记忆

### 方法局限性
- **理论基础**：LSB为何能良好近似离散Boltzmann分布的理论解释尚不完整
- **超参数调优**：σ等超参数需经验性选择，缺乏系统化自动调优策略
- **模型范围**：目前主要验证SRBM，未探索含hidden-hidden连接的Deep BMs
- **温度依赖性**：βeff随模型参数u动态变化，需每轮估计增加计算负担

### 未来工作方向
1. **架构扩展**：探索SAL在Deep BMs、DBNs等更深架构中的应用
2. **采样器泛化**：研究其他并行非MCMC采样器（如量子退火器、CIM）与SAL的结合
3. **理论深化**：阐明LSB的收敛性和近似保证，建立与Langevin Monte Carlo的严格联系
4. **自动化调优**：开发基于元学习或自适应机制的LSB超参数选择策略
5. **CEM分析**：研究CEM的统计性质、收敛条件及影响估计精度的因素

---

**核心贡献**：SAL框架通过LSB并行采样和CEM高效温度估计，**首次在实践意义上解锁了SRBM的表达能力**，为超越RBM的能量基生成模型开辟了新路径。

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
- **缺乏真实训练数据**：系统发育学中无法获得ground truth数据，传统监督学习方法依赖模拟数据但泛化能力有限
- **可扩展性瓶颈**：当taxa数量n>4时，可能的unrooted树数量为(2n-5)!!，导致多分类输出不可行
- **置换不变性缺失**：CNN和ResNet对输入序列顺序敏感，而系统发育推断要求对taxa排列不变
- **计算效率低下**：现有深度学习方法（如Phyloformer）内存占用巨大，难以处理大规模数据

### 提出的新方法
- **基于距离的度量学习框架**：将多序列比对转换为成对距离矩阵，再通过neighbor-joining (NJ)构建树，而非直接预测树结构
- **两类最小化神经网络架构**：
  - **Sequence networks (S)**：将每个序列独立映射到嵌入空间，再计算成对距离（线性空间复杂度O(n)）
  - **Pair networks (P)**：直接处理序列对，更具表达能力（二次空间复杂度O(n²)）
- **置换等变层设计**：利用群论构建对taxa和sites置换不变的神经网络层，无需数据增强
- **理论指导的架构设计**：基于Bourgain定理和Johnson-Lindenstrauss引理，证明低维嵌入可近似树度量空间

### 相比现有方法的优势
- **计算效率高**：模型参数少（最小仅7,657参数），推理速度快，内存占用显著低于Phyloformer
- **强可扩展性**：可处理n>15、L>200bp的大规模系统发育问题
- **置换不变性保证**：通过架构设计而非数据增强实现，训练更稳定
- **泛化能力强**：在适当训练数据下，性能可与state-of-the-art方法（IQ-TREE）媲美
- **灵活性**：可适配多种进化模型（JC, K2P, HKY, LG等）

---

## 2. 核心实验方法和设置

### 数据集
- **模拟数据**（因缺乏真实ground truth）：
  - **树结构**：出生-死亡模型 BD(λ=1, μ=0.5, n=20)，生成20-taxa树
  - **序列模拟**：使用IQ-TREE的AliSim工具
  - **进化模型**：
    - **JC**：Jukes-Cantor模型（简单）
    - **K2P**：Kimura 2参数模型
    - **HKY+Γ**：HKY模型+Gamma分布速率异质性（连续Gamma(1,1)）
    - **LG+Γ**：Le-Gascuel蛋白矩阵+Gamma分布速率异质性
    - **LG+indel**：上述模型+插入缺失（indel率=0.01×替换率）

### 实验设置
- **训练集**：10⁵个比对，每个500个字符位点
- **测试集**：500个比对，每个1000个字符位点（indel数据集为可变长度，最长5000）
- **验证集**：独立的60-taxa数据集用于early stopping
- **硬件**：单张Tesla V100-PCIE (32GB内存)

### 评估指标
- **Robinson-Foulds (RF) 距离**：未加权拓扑差异，归一化到[0,1]
- **模型性能**：Mean RF distance (dRF) 越小越好
- **内存占用**：训练/推理的显存消耗（MB）

### 基线方法对比
- **传统距离方法**：Hamming distance (dH), Jukes-Cantor (dJC), Kimura 2P (dK2P)
- **最大似然法**：IQ-TREE（提供正确模型规范）
- **深度学习方法**：Phyloformer（预训练模型）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（20-taxa树，RF距离）

| 方法 | JC | K2P | HKY | LG | LG+indel |
|------|-----|-----|-----|-----|----------|
| **dH** | 0.1244 | 0.1204 | 0.1032 | 0.0847 | 0.0739 |
| **dJC** | 0.1993 | 0.1475 | 0.0889 | 0.0690 | 0.0646 |
| **dK2P** | 0.2229 | 0.1963 | 0.0893 | - | - |
| **Site-Invariant-S** | 0.1274 | 0.1389 | 0.1018 | 0.0810 | 0.2087* |
| **Full-Invariant-S** | 0.1253 | 0.1527 | 0.0975 | 0.0695 | 0.1382* |
| **Sites-Attention-P** | 0.1269 | 0.1219 | 0.1060 | 0.0661 | 0.0716* |
| **Hybrid-Attention-SP** | **0.1163** | **0.1011** | **0.0856** | **0.0532** | 0.1299* |
| **Full-Attention-SP** | 0.1664 | 0.1020 | 0.0854 | 0.0547 | **0.0558*** |
| **Phyloformer** | - | - | - | 0.0506 | 0.0601 |
| **IQ-TREE** | 0.1732 | 0.1613 | 0.0791 | **0.0438** | **0.0453** |

*注：*表示添加了空间层（CNN或位置编码）*

### 关键发现
1. **模型复杂度匹配**：简单模型（JC/K2P）中，简单网络（Site-Invariant-S）表现最佳；复杂模型（HKY/LG）中，注意力网络（Hybrid/Full-Attention）优势明显
2. **Hybrid-Attention-SP最优**：在多数条件下（除LG+indel外）取得最佳性能，平衡了效率与精度
3. **超越传统距离**：所有训练网络在多数条件下优于dH/dJC/dK2P
4. **与SOTA差距**：IQ-TREE在HKY/LG/LG+indel上仍领先，但差距在缩小（LG上dRF差0.0094）
5. **注意力头数影响**：头数<4时性能急剧下降，与理论结果一致
6. **通道数敏感性**：隐藏通道数<32或>128时验证误差快速上升

### 消融实验结果
- **信息压缩**：Sites-Attention-P将i.i.d.数据的site模式压缩至<2%（DNA）和~15%（蛋白）
- **度量性质**：部分训练网络（如Sites-Attention-P）在未加正则化情况下仍满足三角不等式
- **序列长度 scaling**：性能随序列长度增加而提升（统计一致性）
- **Taxa数量 scaling**：RF误差随taxa数量增加而上升（图2）

---

## 4. 关键结论和发现

### 主要发现
1. **信息共享的悖论**：理论上taxa间信息共享对复杂模型至关重要，但实验显示不共享taxa维度的简单方法（如Site-Invariant-S）表现意外良好。Hybrid-Attention-SP优于Full-Attention-SP，表明**仅需少量taxa间信息**即可达到最佳效果
2. **NJ算法的鲁棒性**：只要局部关系学习正确，长距离误差在NJ迭代中会被平均化，这解释了为何轻量级架构有效
3. **学习距离的隐式先验**：学习到的距离变换（如dJC近似）不仅拟合目标函数，还编码了训练集中树直径的分布等先验信息，性能超越传统最大似然距离
4. **注意力机制的代价**：全注意力网络内存消耗巨大（Full-Attention-SP: 20,399MB），而Hybrid架构更实用（14,562MB）

### 方法局限性
- **长枝吸引问题**：学习距离可能像ML/MP算法一样易受长枝吸引（long-branch attraction）影响
- **串联比对处理差**：预训练网络在串联多基因比对上表现不佳，无法匹配dJC/dK2P
- **模型专用性**：当前为每个进化模型单独训练，实际应用需单一模型适配多种进化过程
- **拓扑vs分支长度**：注意力网络分支长度预测更准，但拓扑精度提升有限，而拓扑才是核心需求

### 未来工作方向
1. **统一模型训练**：开发能学习并区分多种进化模型的单一高容量网络
2. **高效注意力机制**：采用局部注意力、分组查询注意力（GQA）等技术处理百万级位点的串联比对
3. **损失函数优化**：探索基于四重奏一致性（quartet consistency）的对比学习损失，替代RF距离
4. **集成树构建**：将NJ过程融入网络，实现端到端可微分，避免贪心算法的局限性
5. **真实数据验证**：在RAxML-Grove等真实基因树上评估迁移学习能力

---

**代码可用性**：https://github.com/bkrosenz/phyloDNN

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

## 1. 主要贡献和创新点

### 解决的问题
- **核方法计算瓶颈**：传统核方法需要计算核矩阵，复杂度为 $\mathcal{O}(N^3)$，难以处理大规模数据集（$N \approx 10^7-10^9$）
- **超参数搜索效率低**：特征超参数 $\theta$（如Fourier特征的周期性参数）通常依赖交叉验证，需重复求解优化问题 $P$ 次，计算冗余

### 提出的新方法
- **Feature Learning (FL) 模型**：将多个不同超参数 $\theta_p$ 的张量积特征映射 $\phi_{\theta_p}(x)$ 通过**可学习的线性组合**构造新特征：
  $$f(x) = \left[\sum_{p=1}^{P} \lambda_p \phi_{\theta_p}(x)\right]^\top w$$
- **双重CPD结构**：特征映射本身表示为rank-P的Canonical Polyadic Decomposition (CPD)，模型参数 $w$ 也表示为rank-R的CPD，形成嵌套张量网络结构
- **联合优化框架**：通过**Alternating Least Squares (ALS)** 同时学习：
  - 特征权重 $\lambda_p$（衡量各超参数配置的重要性）
  - 模型参数 $w$（CPD cores $W^{(d)}$）

### 相比现有方法的优势
- **计算效率**：训练复杂度从 $\mathcal{O}(I^{2D}[N+I^D])$ 降至 $\mathcal{O}(\mathcal{E}D N I R[P+I R])$，**速度提升3-5倍**
- **避免重复计算**：单次优化替代 $P$ 次交叉验证，利用张量结构共享计算
- **表达能力强**：线性组合特征比单一特征更具表达能力，在Yacht等数据集上性能显著优于CV
- **内存高效**：CPD将参数存储从 $\mathcal{O}(I^D)$ 降至 $\mathcal{O}(DIR)$

---

## 2. 核心实验方法和设置

### 数据集
- **小规模UCI回归数据集**：
  - Airfoil ($N=1502, D=5$)
  - Concrete ($N=1030, D=8$)
  - Energy ($N=768, D=8$)
  - Wine ($N=6497, D=11$)
  - Yacht ($N=308, D=6$)
- **大规模数据集**：Airline（$N=5,929,413, D=8$，590万条航班延误记录）

### 实验设置
- **数据划分**：80%训练，20%测试，重复10次（Airline重复5次）
- **特征**：量化Fourier特征，每维基函数数 $I_d = I$（取值2-64）
- **模型配置**：CPD rank $R$ 和 $I$ 的选择确保模型参数量小于样本量 $N$（欠参数化）
- **训练**：ALS算法运行 $\mathcal{E}=10$ 个epoch
- **硬件**：笔记本（i7-1365U, 16GB RAM）用于小数据集；服务器（2×AMD EPYC 7252, 256GB RAM）用于Airline

### 评估指标
- **预测质量**：测试集均方误差 (MSE)
- **效率**：训练时间（秒）
- **统计显著性**：报告均值和标准差

### 基线方法
- **交叉验证基线 (CV)**：量化CPD核机器 + 6折交叉验证搜索 $\theta$，候选集 $\theta \in \{10.2, 128, 25, 64, 600, 2000, 1024\}$

---

## 3. 主要实验结果和性能指标

### 正则化消融实验（表1）
| 正则化类型 | 关键发现 |
|------------|----------|
| **L1** | **最优综合性能**：MSE略低（如Concrete 0.139），训练最快（Airfoil 2.7秒），因稀疏性减少计算量 |
| L2 | 性能接近L1，但计算稍慢 |
| Fixed Norm (FN) | 无需调$\beta$超参数，但解密集，计算效率中等 |
| **非负约束 (P)** | 对性能影响不一，Yacht数据集上MSE显著恶化（0.112→0.366） |

### FL vs 交叉验证对比（表2）
| 数据集 | MSE (FL) | MSE (CV) | Time (FL) | Time (CV) | **加速比** |
|--------|----------|----------|-----------|-----------|------------|
| Airfoil | 0.184±0.02 | 0.223±0.02 | 3.0s | 23s | **7.7×** |
| Energy | 0.003±0.0 | 0.003±0.0 | 0.91s | 5.5s | **6.0×** |
| Yacht | **0.112±0.02** | 0.358±0.06 | 0.149s | 0.615s | **4.1×** |
| Concrete | 0.139±0.03 | 0.118±0.02 | 1.2s | 8.8s | **7.3×** |
| Wine | 0.692±0.07 | 0.652±0.04 | 33s | 152s | **4.6×** |
| **Airline** | 0.804±0.0 | 0.779±0.0 | **15159s** | **56590s** | **3.7×** |

### 特征数 $P$ 的扩展性（图2）
- **训练时间**：FL模型随 $P$ 增长曲线更平缓，CV呈线性增长
- **MSE稳定性**：FL在多数数据集上MSE标准差更小，鲁棒性更强
- **性能交叉点**：当 $P>4$ 时，FL在Yacht上显著优于CV

---

## 4. 关键结论和发现

### 主要发现
1. **效率与精度兼得**：FL模型在保持预测质量相当的前提下，实现**3-5倍持续加速**，在大规模数据上优势更明显
2. **L1正则化最优**：L1正则化因诱导稀疏性（部分$\lambda_p=0$）进一步加速训练，是推荐配置
3. **表达力优势**：特征线性组合比单特征更灵活，在Yacht等非线性强的数据上MSE降低**68%**
4. **可扩展性**：成功处理590万样本的Airline数据集，验证了其在大规模问题上的实用性

### 方法局限性
- **超参数调优**：仍需调整$\beta$（除Fixed Norm外）和CPD rank $R$
- **非凸优化**：ALS保证局部收敛，结果依赖初始化
- **特征构造**：目前聚焦Fourier特征，其他特征类型需进一步验证
- **理论保证**：缺乏全局最优性理论分析

### 未来工作方向
1. **并行化**：利用求和结构实现维度/特征并行
2. **概率 formulation**：引入贝叶斯框架进行不确定性量化
3. **自适应$P$**：动态调整特征配置数量
4. **分类任务**：当前聚焦回归，需扩展至分类问题

---

**核心贡献总结**：该工作通过张量网络重构，将离散的超参数搜索转化为连续的特征学习问题，在**计算效率**和**模型表达力**之间建立了新的平衡点，为大规模核方法应用提供了实用化解决方案。

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
- **传统优化方法的瓶颈**：MMO游戏数值系统优化依赖大规模在线实验或预定义统计模型，存在**高时间成本**（需数周至数月观察）、**高机会成本**（错误调整可能导致不可逆的经济系统崩溃和玩家流失）、**测试局限性**（重大机制变更无法通过小规模A/B测试验证）三大核心问题
- **离线模拟的保真度不足**：现有简化离线模拟系统无法准确模仿真实玩家的推理过程和对干预措施的复杂反应，缺乏微观层面的玩家行为洞察

### 提出的新方法
- **生成式多智能体模拟系统**：首次构建面向MMO游戏的**可扩展LLM驱动模拟系统**，通过**监督微调(SFT)** 和**强化学习(RL)** 在**大规模真实玩家行为数据**上适配LLM
- **双模块架构**：
  - **Player Agent**：三阶段微调流程（词汇扩展→行为规划SFT→RL增强）实现高保真玩家决策模拟
  - **Battle Server**：基于真实战斗日志训练的数据驱动环境模型，预测多人战斗结果和收益
- **系统化干预框架**：支持实时参数调整和干预效果监测的图形化实验管理界面

### 相比现有方法的优势
- **高保真度**：在真实玩家轨迹预测上比未微调模型提升**10.19%**准确率，能准确复现战斗结果、物品购买和对干预的因果响应
- **可解释性**：通过智能体的详细行为和推理分析提供可解释洞察，突破传统统计模型的"黑盒"限制
- **成本效益**：提供可靠、可解释且成本高效的数值设计框架，避免直接在线实验的风险
- **全面性**：首次实现**完整游戏系统模拟**，而非孤立场景（如仅关注经济谈判或Pay-to-Win机制），捕捉玩家行为的连锁后果

---

## 2. 核心实验方法和设置

### 数据集
- **真实游戏数据**：数百万条真实玩家记录（登录/登出、战斗、购买、社交互动），用于构建玩家画像和校准模型
- **战斗数据**：2025 S1赛季连续比赛日志，覆盖多赛季数据，用于训练Battle Server
- **评估数据集**：10,000条单日游戏轨迹，涵盖多样化玩家画像

### 实验设置
- **基础模型**：Qwen2.5-1.5B
- **微调配置**：LoRA适配器（rank=16, α=0.2）
- **玩家画像**：基于十余个游戏内特征（游戏场次、声望、段位、游戏时长、模式偏好、平均击杀数等）聚类出**五类典型玩家**：
  1. 稳定发展型玩家（Stable Development）
  2. 新手玩家（Novice）
  3. 财富积累精英玩家（Wealth-Accumulating Elite）
  4. 休闲玩家（Casual）
  5. 高技能玩家（High-skill）

### 评估指标
- **Player Agent**：下一步行动预测准确率（四分类：offline/battle/buy/sell）
- **Battle Server**：胜负预测准确率、每场收入预测误差
- **干预评估**：行为分布变化率（如非正式交易比例）
- **系统级指标**：财富分布、段位分布、资源消耗、活跃度等宏观统计量

### 基线方法
- **DeepSeek-V3**：未经过游戏领域微调的SOTA LLM作为baseline

---

## 3. 主要实验结果和性能指标

### Player Agent性能
- **整体准确率提升**：相比DeepSeek-V3基线，微调后模型实现 **+10.19%** 的绝对准确率提升
  - 仅SFT微调：**+8.34%** 提升
  - 加入用户画像信息后：**额外+1.85%** 提升
- **行为分布对齐**：模拟的玩家行为分布（战斗、购买、出售、离线）与真实数据高度一致（图4a）

### Battle Server性能
- **预测准确性**：对2025 S2赛季数据的预测结果显示：
  - **财富积累精英玩家**和**稳定发展型玩家**：胜负率和收入预测**高度准确**
  - **新手**和**休闲玩家群体**：预测波动较大，误差相对较高（归因于行为不一致性）
- **跨赛季泛化**：在严格无数据泄露的S2赛季验证中，模型保持了稳定的预测能力

### 干预案例研究结果
- **黑市引入干预**：模拟了从非正式交易到官方交易平台的机制变更
- **行为迁移效果**：非正式游戏内交易比例从 **27.4% → 1.5%** （下降94.5%）
- **因果响应保真度**：模拟智能体能识别新交易渠道并相应调整行为，仅少数因干预前习惯持续使用旧机制

### 系统级验证
- **宏观统计一致性**：数百个智能体模拟数周的游戏过程，财富分布、段位分布等宏观指标与真实世界**强一致**
- **微观行为可观测性**：支持在每个时间步对单个智能体的状态、属性和历史进行细粒度分析

---

## 4. 关键结论和发现

### 主要发现
1. **数据驱动适配的有效性**：通过SFT和RL在真实行为数据上微调，可将通用LLM转化为**领域专家**，显著提升决策预测准确性
2. **分层模拟架构的可行性**：Player Agent（决策）与Battle Server（环境反馈）解耦的架构能有效模拟复杂MMO动态，且**可扩展**
3. **干预因果效应的可预测性**：系统能**忠实复现**真实世界机制变更的因果效应，为预部署设计决策提供可靠验证
4. **玩家异质性的重要性**：不同玩家群体（尤其是精英vs新手）对相同机制的反应存在显著差异，系统能捕捉这种**异质性**

### 方法局限性
- **新手/休闲玩家建模挑战**：这两类玩家行为波动大、一致性低，导致预测误差较高
- **计算资源需求**：大规模多智能体模拟需要大量LLM推理计算，Resource Manager虽通过通信池优化但仍存在扩展瓶颈
- **游戏类型特定性**：当前系统针对**撤离射击类MMO**（如Escape from Tarkov）设计，通用性需进一步验证
- **长期依赖建模**：未明确评估对长期（数月至数年）游戏动态的模拟能力

### 未来工作方向
- **增强通用性**：探索系统在更多MMO子类型（如MMORPG、MOBA）中的应用
- **提升效率**：优化多智能体并行推理架构，降低模拟成本
- **长期模拟**：验证系统在更长模拟周期下的稳定性和准确性
- **自动化设计**：结合优化算法实现**自动化的数值参数搜索和机制设计**
- **社交行为深化**：增强智能体间复杂社交互动（如公会、欺诈、合作）的建模能力

---

**核心价值**：该工作首次证明了**LLM驱动的生成式多智能体系统**可作为MMO游戏数值设计的**可靠数字孪生**，在保真度、可解释性和成本效益上全面超越传统方法，为游戏行业提供了**从"测试-观察"到"模拟-预测"**的范式转变。

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
# Dynamic Configuration of On-Street Parking Spaces using Multi Agent Reinforcement Learning - 核心总结

## 1. 主要贡献和创新点

### **解决的问题**
- **动态路边停车位配置问题**：传统静态clearways无法适应实时交通变化，而固定分配停车位会加剧拥堵。论文首次从基础设施视角出发，将停车位配置建模为优化问题，目标是最小化**平均行驶时间损失**和**步行距离**的加权和。

### **提出的新方法**
- **双层多智能体强化学习框架**：
  - **车道级智能体（Lane Level RL Agents）**：部署在每个车道，使用**改进的Deep Q-learning**决定清除的停车位数量（0到最大值）
  - **街区级智能体（Block Level Agents）**：监督车道级智能体，确保街区整体停车供需平衡
  
- **创新的DQN架构**：融合**LSTM**和**Graph Attention Networks (GAT)** 捕获时空相关性
  - **LSTM**：捕获本车道历史交通拥堵时序模式
  - **GAT**：建模相邻车道的时空依赖关系
  - **全连接网络**：处理当前观测数据

### **相比现有方法的优势**
- **主动预测性**：能预判未来交通状况，提前调整配置（区别于传统被动响应）
- **高可扩展性**：分层架构适合大规模路网，采用集中训练-分布执行范式
- **显著性能提升**：相比clearways等静态方案，**行驶时间损失最高减少47.79%**，步行距离增加**不足2米**
- **环境效益**：车辆排放（CO₂, CO, HC, PMx, NOx）最高减少40%

---

## 2. 核心实验方法和设置

### **数据集**
- **真实世界数据**：澳大利亚墨尔本市中心区域
  - 3.5km × 2km区域，15个信号交叉口，38个路段，**3042个路边停车位**
  - 71,217辆车的全天轨迹（6am-9pm）
  - 数据源：OpenStreetMap、SCATS感应线圈数据、停车传感器数据
  
- **合成数据**：网格网络，用于**敏感性分析**
  - 参数范围：车辆插入率(60-110 veh/s)、停车概率(0.1-0.4)、停车时长(600-3600s)、网格大小(3×3到7×7)

### **实验设置**
- **仿真平台**：SUMO微观交通仿真器 + TraCI Python API
- **训练框架**：PyTorch，Adam优化器，MSE损失
- **网络参数**：
  - 全连接层：2→32维
  - LSTM：2层，32维隐状态，序列长度10
  - GAT：2个注意力头，64维隐状态
  - Q网络：3层全连接(96→128→3)

### **评估指标**
- **平均时间损失** ($t_{loss}$)：实际行驶时间 - 自由流时间
- **时间损失减少百分比** ($t_{loss}\%$)：相对于No-PA基准的改善幅度
- **平均步行距离** ($d_{walk}$)：目的地与实际停车位置距离
- **车辆排放**：CO₂, CO, HC, PMx, NOx

### **基线方法**
- **No-PA**：不限制任何停车位
- **C-PA**：高峰时段clearways（静态方案）
- **S-PA**：静态清除交叉口附近固定数量停车位
- **D-PA变体**：PPO, A2C, Double DQN, Dueling DQN（动态配置但网络架构不同）

---

## 3. 主要实验结果和性能指标

### **真实世界数据核心结果**
| 停车概率 | 方法 | $t_{loss}$↓ (s) | $t_{loss}\%$↑ | $d_{walk}$↑ (m) |
|---------|------|----------------|--------------|----------------|
| 0.4 | **Ours** | **121.22** | **47.79%** | 3.52 |
| 0.4 | D-PA (DQN) | 155.71 | 32.93% | 2.72 |
| 0.4 | C-PA | 198.55 | 14.48% | 70.70 |
| 0.4 | No-PA | 232.17 | - | 2.12 |

**关键发现**：
- **所有停车概率下均最优**：在停车概率0.1-0.4范围内，时间损失减少**13.03%-47.79%**
- **步行距离代价极小**：仅增加1.44-3.52米（<2米阈值）
- **显著优于clearways**：C-PA虽减少时间损失，但步行距离激增60-70米

### **敏感性分析结果**
- **交通量增加时优势更显著**：车辆插入率110 veh/s时，性能提升幅度最大
- **停车需求越高效果越好**：停车概率0.4时减少47.79%，而0.1时仅13.03%
- **网络规模可扩展**：在7×7网格中仍保持最优性能

### **消融实验结果**
| 模型架构 | 停车概率0.4时的$t_{loss}\%$ |
|---------|---------------------------|
| **DQN + LSTM + GAT** | **47.79%** |
| DQN + LSTM | 38.52% |
| Vanilla DQN | 32.93% |

**结论**：LSTM捕获时序信息带来**~6%**提升，GAT捕获空间依赖带来额外**~9%**提升

### **环境效益**
- **排放减少**：CO₂, CO, HC, PMx, NOx最高减少**40%**
- **全天改善**：如图6所示，早晚高峰时段改善最显著

---

## 4. 关键结论和发现

### **主要发现**
1. **动态配置远优于静态方案**：自适应调整停车位能平衡交通效率与停车需求，而clearways等静态方案牺牲停车便利性却收益有限
2. **时空建模至关重要**：LSTM+GAT架构有效捕获了交通流的动态演化和车道间耦合，使智能体能**提前预判拥堵**并主动清除关键停车位
3. **分层架构的有效性**：街区级智能体成功防止了车道级智能体过度清除停车位，确保停车可用性不低于阈值（80%）

### **方法局限性**
- **依赖V2I技术**：需要实时获取交通状态和下发配置指令，对基础设施有要求
- **训练成本**：DQN需要大量仿真数据训练，迁移到新城网可能需要微调
- **简化假设**：未考虑驾驶员行为不确定性、违规停车等现实复杂因素

### **未来工作方向**
- **自动驾驶场景扩展**：在混合交通（有人驾驶+自动驾驶）中优化配置
- **多目标强化学习**：显式处理旅行时间、步行距离、排放等多目标权衡
- **实时部署研究**：解决监管合规性和社会接受度等实际部署挑战

---

**总结**：该论文提出的双层MARL框架通过创新的DQN架构，在保持停车便利性的前提下，显著提升了城市交通效率，为智能交通系统提供了可扩展的动态停车位管理解决方案。

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
- **LLM智能体规模化部署成本过高**：执行新型LLM智能体时，高容量模型推理成本高昂，阻碍快速原型验证和规模化应用
- **传统成本降低方法摩擦成本高**：
  - **微调**：需要长训练周期、大量数据、ML专业知识，且需为每个新任务重复进行
  - **提示工程**：需要大量人工试错，开发缓慢且脆弱

### 提出的新方法
**上下文蒸馏（In-Context Distillation）+ 自洽级联（Self-Consistency Cascades）**
1. **上下文蒸馏**：将知识蒸馏思想适配到in-context learning场景
   - 离线阶段：收集教师模型在少量任务上的完整推理轨迹（观察、推理链、动作）
   - 推理阶段：在每个决策点动态检索最相关的教师示范作为in-context示例，注入学生模型prompt
   - 学生模型（冻结权重）通过模仿教师行为完成知识转移，无需参数更新

2. **自洽级联**：将自洽性作为内省信号判断学生模型置信度
   - 每个步骤采样学生模型N次（N=3）
   - 若输出一致 → 学生可信，执行其动作
   - 若输出分歧 → 学生不确定，回退到教师模型
   - 实现细粒度（单步级别）的自适应路由

### 相比现有方法的优势
- **无需训练**：完全规避微调的计算开销和工程复杂度
- **快速部署**：仅需教师示范和向量索引，可立即应用于新任务
- **成本效益显著**：在保持教师级性能的同时实现2.5-3.5倍成本降低
- **敏捷开发**：支持冻结模型的快速迭代，适合长尾应用场景
- **样本高效**：仅需数百个示范即可有效转移知识

---

## 2. 核心实验方法和设置

### 数据集
1. **ALFWorld**：多步具身推理基准测试
   - 任务：在文本环境中执行动作序列完成目标（如"把苹果放在冰箱里"）
   - 示范集：500个训练任务
   - 测试集：134个分布外评估任务

2. **AppWorld**：复杂API工作流自动化基准
   - 任务：多步API调用组合（Gmail、Calendar、Contacts等）
   - 示范集：147个训练/验证任务
   - 测试集：168个测试任务
   - 通过单元测试验证状态变化和执行轨迹

### 实验设置
- **教师模型**：Claude Sonnet 4.5（高能力）
- **学生模型**：GPT-4.1-mini（低成本，便宜10-100倍）
- **检索模型**：MiniLM-L6-v2（计算embedding相似度）
- **温度参数**：默认0.1（除消融实验外）

### 评估指标
- **准确率**：任务成功完成的比例
- **成本**：每episode的LLM调用总费用（输入+输出token）
  - 按2025年10月API定价计算
  - 归一化为教师成本的相对值
- **教师调用比例**：级联中实际使用教师模型的步骤占比

### 基线方法对比
1. **Teacher**：仅教师模型（准确率上限）
2. **Student (ZS)**：学生模型零样本（成本下限）
3. **Student (IC)**：仅上下文蒸馏（无级联）
4. **Student (Cascade only)**：仅自洽级联（无示范）
5. **Random Mix**：固定比例随机混合师生调用
6. **GPT-4.1 (ZS)**：中等规模模型零样本
7. **Llama-3.3-70B**：开源模型验证通用性

---

## 3. 主要实验结果和性能指标

### 核心性能数据

| 方法 | ALFWorld 准确率 | ALFWorld 相对成本 | AppWorld 准确率 | AppWorld 相对成本 |
|------|----------------|-------------------|----------------|-------------------|
| Teacher | 0.89 | 1.00 ($0.059) | 0.83 | 1.00 ($0.589) |
| Student (ZS) | 0.18 | 0.31 | 0.28 | 0.087 |
| **Student (IC)** | **0.87** | **0.43** | **0.55** | **0.15** |
| **Student (IC+Cascade)** | **0.96** | **0.42** | **0.66** | **0.29** |

### 关键发现

**1. 成本-准确率权衡突破**
- **ALFWorld**：在**等准确率**下实现**2.5倍成本降低**（$0.059 → $0.024），且准确率**超越教师**（96% vs 89%）
- **AppWorld**：在**等准确率**下实现**2倍成本降低**，以29%成本恢复79%教师性能
- 相比随机混合基线，在同等成本下提升成功率12-28个百分点

**2. 上下文蒸馏效果显著**
- 单独使用上下文蒸馏即可缩小师生差距：
  - ALFWorld：准确率从0.18→0.87（97%教师性能），成本仅43%
  - AppWorld：准确率从0.28→0.55，成本仅15%
- 学生模型在示范引导下轨迹更短（ALFWorld：27→12步，AppWorld：19→12步）

**3. 自洽级联的智能路由**
- 教师调用比例极低：ALFWorld仅**3.9%**步骤，AppWorld **22.2%**步骤
- 采样开销可忽略：学生模型便宜10-100倍，N次采样成本仍远低于单次教师调用
- 有效识别不确定性：在示范不匹配或状态模糊时自动回退

**4. 跨模型通用性**
- **Llama-3.3-70B**作为学生模型：
  - ALFWorld：准确率从0.50→0.87（IC）→0.93（IC+级联）
  - AppWorld：准确率从0.11→0.32（IC）→0.44（IC+级联）
- 证明方法不依赖特定API，适用于开源模型

### 消融实验结果

**1. 检索示例数量k的影响**
- **ALFWorld**：k=1→4时准确率从0.75→0.86，k>6后边际收益递减（成本增加76%仅提升1.5%）
- **AppWorld**：k=5时达到峰值0.57，k>5无一致收益
- **最优配置**：ALFWorld k=6，AppWorld k=3（默认设置）

**2. 教师数据库规模**
- **ALFWorld**：
  - 20个示范：0.58准确率（已远超零样本0.18）
  - 100个示范：0.836准确率（94%教师性能）
  - 500个示范：0.87准确率（98%教师性能）
- **AppWorld**：147个示范达到0.548准确率（67%教师性能）
- **数据效率**：极小数据库即可显著超越零样本基线

**3. 检索粒度对比**
- **单步检索 vs 全程检索**：准确率相同（ALFWorld 0.87），但**单步检索成本低26%**
  - 原因：全程检索包含完整轨迹（10-20步），而单步检索仅选择最相关的3-5步窗口
  - 结论：检索相关性比上下文体积更重要

**4. 任务难度分层分析（AppWorld）**
| 难度 | 教师 | Student (IC+Cascade) | Student (ZS) | 任务数 |
|------|------|----------------------|--------------|--------|
| 1（简单） | 0.96 | **0.91** (95%恢复) | 0.51 | 57 |
| 2（中等） | 0.85 | **0.70** (82%恢复) | 0.29 | 48 |
| 3（困难） | 0.71 | **0.43** (61%恢复) | 0.06 | 63 |

- 在困难任务上性能差距扩大，但上下文蒸馏仍提供37-40个百分点的绝对提升

---

## 4. 关键结论和发现

### 主要发现
1. **上下文蒸馏是高效的非参数知识转移机制**：通过动态检索教师示范，冻结的学生模型可达到接近教师的性能，无需昂贵的微调
2. **自洽性是有效的内省信号**：采样一致性完美捕捉in-context学习效果，实现细粒度自适应路由
3. **成本回收极快**：ALFWorld在**843个任务**后回本，AppWorld仅需**207个任务**；规模化部署（100万任务）可分别节省 **$34,900** 和 **$419,000**
4. **方法具有普适性**：适用于不同模型架构（GPT-4.1-mini、Llama-3.3-70B）和不同领域（具身推理、API编排）

### 方法局限性
1. **困难任务性能下降**：在高度复杂的任务（AppWorld难度3）上，学生模型仅恢复61%教师性能
2. **依赖示范质量**：检索相关性是关键，当测试任务与示范差异显著时效果减弱
3. **教师模型初始成本**：需要一次性收集教师轨迹，尽管成本可快速摊销
4. **动作空间假设**：自洽性检查在存在多种等价动作的任务上需要辅助验证器（如AppWorld的代码等价性判断）

### 未来工作方向
1. **扩展数据库规模**：在AppWorld上探索超过147个示范的效果
2. **优化检索策略**：研究更智能的示例选择机制，如多样性加权或难度感知检索
3. **缓存感知成本模型**：利用KV缓存减少重复输入token成本
4. **难度感知路由**：结合任务元数据（如难度标签）设计混合路由策略，在成本与准确率间灵活权衡
5. **多教师蒸馏**：整合多个专业教师模型的示范，提升学生模型泛化能力
6. **在线示范更新**：探索在部署过程中动态更新示范数据库，适应分布变化

### 实践意义
该方法为**敏捷开发**提供了理想解决方案：
- **适合场景**：快速原型验证、长尾工作流自动化、中等规模部署（数千至数百万任务）
- **不适合场景**：对准确率要求极高且成本不敏感的稳定生产系统（仍建议微调）
- **工程优势**：无需ML基础设施，仅需向量数据库和示范收集，大幅降低技术门槛

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
# 论文总结：Adapting Tensor Kernel Machines to Enable Efficient Transfer Learning for Seizure Detection

## 1. 主要贡献和创新点

### 解决的问题
- **医疗数据稀缺性**：癫痫检测中患者间变异性大，但患者特定数据收集和标注成本高昂
- **传统迁移学习低效**：现有Adaptive SVM方法在适应新患者时会增加模型参数（增加支持向量），导致模型膨胀
- **资源约束挑战**：可穿戴设备需要轻量级模型以实现实时推理和边缘计算

### 提出的新方法
- **Adaptive Tensor Kernel Machine (Adapt-TKM)**：将Adaptive SVM的思想与张量核机器（TKM）结合
- **核心机制**：通过正则化项约束适应模型与源模型的权重距离，而非直接存储源数据或支持向量
- **技术实现**：使用CPD（Canonical Polyadic Decomposition）低秩张量网络在原始空间（primal domain）学习紧凑表示

### 相比现有方法的优势
- **参数效率**：比Adaptive SVM少约100倍参数（$1.99\times10^4$ vs $1.36\times10^6$）
- **推理速度**：快约100倍（$3.9\times10^{-5}$秒 vs $3.3\times10^{-3}$秒/样本）
- **隐私保护**：不存储源患者数据，仅传递模型权重
- **模型稳定性**：适应过程不增加模型复杂度，适合资源受限设备

---

## 2. 核心实验方法和设置

### 数据集
- **SeizeIT1**：耳后EEG数据集，包含医院术前评估期间采集的多天监测数据
- **数据特点**：89%为局灶性意识障碍性发作（FIA），91%起源于（额）颞叶
- **预处理**：1-25Hz带通滤波，2秒分段（训练数据：发作段90%重叠，非发作段无重叠）

### 实验设置
- **三种模型对比**：
  - **PI（Patient-Independent）**：患者独立模型，留一患者交叉验证
  - **PS（Patient-Specific）**：患者特定模型，仅使用目标患者数据
  - **PA（Patient-Adapted）**：患者适应模型，用少量目标数据微调PI模型

- **特征工程**：21个特征/通道（时域、频域、熵），Yeo-Johnson变换，缩放至[-0.5, 0.5]

### 评估指标
- **事件级评估**：采用any-overlap方法，预测与真实发作有任何重叠即视为正确
- **核心指标**：Sensitivity（灵敏度）、Precision（精确率）、F1-score、False Alarms/24hr（每日误报率）
- **后处理**：仅当连续10段中至少8段为阳性时才触发警报（最小发作持续时间10秒）

### 基线方法
- SVM（RBF核）和Adaptive SVM
- Tensor Kernel Ridge Regression（TKRR）及其变体
- 超参数：特征映射参数（$M=14, U=1.75, \sigma=\sqrt{1/10}$），CPD秩$R=4$，$\mu=0.01$

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

**关键发现**：TKRR-PA在F1-score上比TKRR-PI提升7.6%，比SVM-PA提升8.1%，同时误报率降低56.7%

### 模型效率对比（表V）
| Model | 参数数量 | 推理时间 |
|-------|----------|----------|
| SVM-PI/PA | $1.363\times10^6$ | $3.3\times10^{-3}$秒 |
| TKRR-PI/PS/PA | **$1.988\times10^4$** | **$3.9\times10^{-5}$秒** |
| SVM-PS | $1.140\times10^4$ | $3.3\times10^{-5}$秒 |

**效率提升**：TKRR系列模型参数减少**98.5%**，推理速度提升**84.6倍**

### 消融实验结果
- **正则化参数μ的影响**（图6）：不同患者最优μ差异显著
  - 高μ（0.1）：适合PI模型已表现良好的患者（如33, 65, 78号）
  - 中μ（0.01）：平衡源-目标知识，整体最优
  - 低μ（$10^{-4}$）：适合PS模型相对更好的患者

- **初始化策略**（图7）：用源模型权重初始化使收敛速度提升约2.6倍（42轮 vs 110轮达最优）

- **个体患者分析**（图5）：15名患者中，**9名**通过PA模型获得性能提升，但**5名**出现性能下降

---

## 4. 关键结论和发现

### 主要发现
1. **有效性验证**：Adapt-TKM能成功个性化患者独立模型，在多数情况下超越PI和PS模型
2. **效率革命**：在保持相当准确率的前提下，实现数量级的参数和速度优化，适合可穿戴设备
3. **隐私优势**：无需存储源患者数据，比Adaptive SVM更符合医疗隐私要求
4. **参数敏感性**：μ的选择对性能影响显著，需根据患者个体特性调整

### 方法局限性
1. **非普适性**：对PI模型已表现优异的患者（F1-score > 0.6），适应可能带来负面效果
2. **超参数选择困难**：
   - μ无明确选择准则，需交叉验证或启发式方法
   - 特征映射参数（$M, U$）和CPD秩$R$增加调参复杂度
3. **数据依赖**：需要一定量的目标患者数据（至少5次发作用于测试）
4. **继承风险**：PA/PS模型继承PI模型的超参数可能非最优

### 未来工作方向
1. **自适应正则化**：开发基于数据相似度或模型置信度的μ自动选择策略
2. **动态秩调整**：研究在适应过程中动态调整CPD秩$R$的方法，避免过度参数化
3. **理论分析**：深入理解何时/为何适应会失败，建立患者选择标准
4. **在线适应**：探索设备端持续学习，实现真正的实时个性化
5. **其他张量网络**：尝试Tensor-Train等结构，可能提供更好的效率-精度权衡

---

**总体评价**：该工作成功将张量网络与迁移学习结合，为资源受限的医疗AI应用提供了高效、隐私友好的解决方案，但在临床部署前需解决超参数自动调优和个体化策略问题。

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
# 论文总结：Beyond Confidence: Adaptive and Coherent Decoding for Diffusion Language Models

## 1. 论文的主要贡献和创新点

### **解决的问题**
- **局部最优陷阱**：现有DLM推理方法依赖单步置信度（如entropy或max probability），易受局部最优影响，且现代LLM存在过度自信问题
- **缺乏理论保证**：当前采样过程缺乏与采样错误率的直接理论联系，难以实现可控的性能保证
- **预算分配低效**：固定均匀解码预算无法适应不同生成场景（如模板化短语 vs 复杂推理步骤）

### **提出的新方法**
**Coherent Contextual Decoding (CCD)** 框架包含两大核心创新：

1. **轨迹修正机制（Trajectory Rectification）**
   - 通过历史上下文增强序列一致性，实现次优路径的早期拒绝
   - 理论等价于通过**条件互信息**（conditional mutual information）建模历史步骤的一致性
   - 近似目标分布：$\overline{p}(x_i|\mathbf{s}) \approx \frac{1}{T-t+1}\sum_{k=0}^{T-t}\hat{p}_\theta(x_i|\mathbf{c}_{T-k,i},\mathbf{s})$

2. **自适应采样策略（Adaptive Sampling Budget）**
   - 动态调整每步的unmasking预算，替代传统的固定扩散步骤
   - 基于一致性指标自动分配预算：上下文敏感token获得小预算，上下文不敏感区域获得大预算

### **相比现有方法的优势**
- **理论最优性**：通过互信息项$I(x_i;\mathbf{c}|\mathbf{s})$显式控制采样误差上界
- **计算高效**：利用解码过程中已计算的预测分布，无需额外前向传播
- **内存友好**：滑动窗口历史缓冲区将内存复杂度从$O(T \times N \times |\mathbb{X}|)$降至$O(d \times V \times |\mathbb{X}|)$
- **即插即用**：无需训练，兼容现有推理优化技术（如remasking、KV-cache加速）

---

## 2. 核心实验方法和设置

### **基础模型**
- **LLaDA-8B-Instruct**（d=2）
- **Dream-7B-Instruct**（d=3）

### **评估数据集**
| 类别 | 数据集 | 样本数 | 特点 |
|------|--------|--------|------|
| 数学推理 | GSM8K | - | 小学数学问题 |
| 数学推理 | MATH | - | 竞赛级数学问题 |
| 代码生成 | HumanEval | 164 | Python函数生成 |
| 代码生成 | MBPP | - | 多步编程问题 |
| 规划 | Trip Plan | 1,600 | 多城市旅行规划 |

### **实验设置**
- **解码预算**：默认$b_t=1$（每步解码1个token）
- **历史缓冲区**：保留最近$d$步的top-V置信token（V=4）
- **评估指标**：
  - **性能**：准确率（accuracy）
  - **效率**：平均解码步数（diffusion steps），作为推理延迟的代理
- **基线方法**：原始论文的标准采样过程（Dream用负entropy，LLaDA用max probability）
- **零样本设置**：除Trip Plan外均采用zero-shot评估

---

## 3. 主要实验结果和性能指标

### **性能提升（CCD固定预算）**
| 模型 | 数据集 | 基线分数 | CCD分数 | 提升 |
|------|--------|----------|---------|------|
| Dream-7B | HumanEval | 52.66 | 57.31 | **+4.65** |
| Dream-7B | Trip Plan | 15.10 | 16.93 | **+1.83** |
| LLaDA-8B | GSM8K | 74.30 | 75.30 | +1.00 |
| LLaDA-8B | HumanEval | 36.50 | 38.41 | +1.91 |

### **加速与性能双提升（CCD-DS自适应预算）**
| 模型 | 数据集 | 加速比 | 步数减少 | 性能变化 |
|------|--------|--------|----------|----------|
| **Dream-7B** | MBPP | **3.78×** | 753.8→270.2 | 58.00→58.00（持平） |
| **Dream-7B** | Trip Plan | **3.48×** | 256→75.2 | 15.10→**19.01**（+3.91） |
| **Dream-7B** | HumanEval | 3.04× | 768→253.2 | 52.66→56.71（+4.05） |
| **LLaDA-8B** | Trip Plan | 2.27× | 256→112.5 | 10.40→11.50（+1.10） |
| **LLaDA-8B** | GSM8K | 1.31× | 512→393.0 | 74.30→75.22（+0.92） |

### **消融实验结果**

**缓冲区大小分析（Dream on Trip Plan）**
- **最优平衡点**：buffer size = 4
  - 准确率峰值：70%（比基线58%提升**20.7%**）
  - 步数减少：256→95.54（**62.7%** reduction）
- 所有buffer size（1-6）均优于基线，证明方法鲁棒性

**温度系数鲁棒性（HumanEval）**
- 在temperature∈{0, 0.1, 0.4, 0.7, 1.0}范围内，CCD-DS**全面优于**基线
- 最大提升：temperature=0时**+9.8%**，temperature=0.7时**+9.0%**

---

## 4. 关键结论和发现

### **主要发现**
1. **语义优先解码**：CCD能区分**语法流畅性**与**语义重要性**，优先解码决定推理方向的关键token（如"Karen"而非"so"），避免级联错误
2. **自动延迟机制**：对上下文敏感的模糊token（如数字、日期），CCD自动推迟解码，利用后续建立的完整上下文提高准确性
3. **EOS生成瓶颈**：DLM存在连续的EOS token生成平台期，CCD-DS通过动态增大预算快速通过，显著减少冗余计算
4. **理论-实践统一**：条件互信息不仅提供理论保证，还自然形成**难度感知调度**——低依赖token早解码，高依赖token晚解码

### **方法局限性**
- **超参数依赖**：需为不同模型调优历史长度$d$（Dream:3, LLaDA:2）和top-V值
- **阈值启发式**：CCD-DS的entropy阈值ε通过稳定性启发式近似，可能非最优
- **长序列挑战**：尽管内存优化，极长序列（N极大）仍可能面临内存压力

### **未来工作方向**
- 探索**自适应$d$和V**策略，根据生成内容动态调整缓冲区
- 结合**speculative decoding**进一步提升速度
- 将CCD思想扩展到**多模态扩散模型**
- 研究**任务感知**的预算分配策略，针对不同任务类型优化调度

---

## **核心贡献总结**
CCD框架通过**历史上下文聚合**和**自适应预算分配**，首次在DLM推理中实现了**速度-质量同步提升**，为扩散语言模型的实际部署提供了理论坚实且高效可行的解决方案。

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
# CREST论文核心结论与实验结果总结

## 1. 主要贡献和创新点

### 解决的问题
- **低资源语言安全覆盖不足**：现有LLM安全护栏主要面向英语等10-20种高资源语言，全球绝大多数低资源语言缺乏有效保护，导致相同有害查询在低资源语言中更容易绕过安全机制
- **计算成本过高**：现有基于LLM的护栏模型（如LlamaGuard3-8B）参数庞大、推理缓慢，难以在边缘设备部署
- **数据稀缺性**：低资源语言缺乏高质量的安全训练数据和标注资源

### 提出的新方法
**CREST (CRoss-lingual Efficient Safety Transfer)**：基于聚类引导的跨语言迁移框架
- **语言聚类策略**：利用XLM-R编码器的表示空间，将100种语言按语义和结构相似性聚为8个簇，选择13种高资源语言作为每个簇的代表进行训练
- **参数高效架构**：基于XLM-RoBERTa构建，仅0.5B参数（Large版）或0.25B参数（Base版）
- **零样本迁移机制**：通过簇内语言表示相似性，实现从训练语言到同簇低资源语言的性能泛化

### 相比现有方法的优势
- **语言覆盖广**：支持100种语言，远超PolyGuard（17种）和DuoGuard（29种）
- **参数效率高**：模型尺寸仅为大型护栏的1/5-1/16，推理速度快10倍以上
- **数据效率高**：无需低资源语言训练数据，仅在13种高资源语言上训练
- **性能均衡**：在高资源和低资源语言上均保持强劲表现，避免"多语言诅咒"

---

## 2. 核心实验方法和设置

### 数据集
**训练数据**：
- **Aegis-AI-Content-Safety-Dataset-2.0**：30k训练样本，覆盖12大风险类别（仇恨、暴力、隐私泄露等）
- 通过高质量机器翻译系统将数据集翻译成13种训练语言

**评估基准**（6个核心数据集）：
- **Aegis-CS2**：内容安全分类
- **HarmBench**：自动化红队测试
- **Redteam2k**：越狱攻击评估
- **JBB-Behaviors & JBB-Judge**：越狱行为判断
- **StrongReject**：空越狱检测
- **CSRT**：代码切换红队测试（多语言混合场景）

### 实验设置
- **训练语言（In-Domain）**：13种高资源语言
  - 西班牙语、英语、德语、俄语、捷克语、芬兰语、印地语、泰米尔语、中文、越南语、阿拉伯语、斯瓦希里语、菲律宾语
  
- **评估语言**：
  - **ID语言**：12种训练语言
  - **OOD低资源语言**：11种（加利西亚语、冰岛语、南非荷兰语、斯洛文尼亚语、僧伽罗语、泰语、马拉地语、普什图语、爪哇语、豪萨语、格鲁吉亚语）

- **模型变体**：
  - CREST-BASE：XLM-RoBERTa-Base（279M参数）
  - CREST-LARGE：XLM-RoBERTa-Large（560M参数）

### 基线方法
| 模型 | 规模 | 支持语言数 | 特点 |
|------|------|------------|------|
| Aegis-Defensive | 7B | 1（英语） | 英语专用 |
| LlamaGuard3 | 8B | 8 | 大规模多语言 |
| PolyGuard-Qwen | 2.5B | 17 | 当前SOTA |
| WalledGuard-C | 0.5B | 1（英语） | 小规模英语 |
| DuoGuard-0.5B | 0.5B | 29 | 两玩家RL训练 |
| PG-Qwen-Smol | 0.5B | 17 | 小规模多语言 |

### 评估指标
- **F1-score**（不安全类别）：主要评估指标
- **跨语言迁移性能**：ID vs OOD语言对比
- **代码切换鲁棒性**：CSRT基准性能
- **文化适应性**：IndicSafe和Cultural Kaleidoscope数据集

---

## 3. 主要实验结果和性能指标

### 英语基准性能
CREST-LARGE在多个基准上**匹配或超越**大规模护栏：
- **HarmBench**：86.87 F1（优于LlamaGuard3的98.09，但低于PG-Qwen的99.66）
- **StrongReject**：96.36 F1（接近SOTA）
- **Redteam2k**：82.80 F1（超越LlamaGuard3的70.84）
- **Aegis-CS2**：85.54 F1（超越LlamaGuard3的76.29）

**关键发现**：尽管参数少5-16倍，CREST-LARGE在英语上表现极具竞争力，验证了其架构有效性。

### 多语言性能对比
**高资源语言（法语、意大利语、德语、葡萄牙语、西班牙语）**：
- CREST-LARGE平均F1达**85.55-86.08**，**全面超越**所有基线
- 特别值得注意的是，法语、意大利语、葡萄牙语**未在训练集中**，体现强大零样本能力

| 模型 | 法语 | 意大利语 | 德语 | 葡萄牙语 | 西班牙语 |
|------|------|----------|------|----------|----------|
| DuoGuard | 73.81 | 61.42 | 76.86 | 67.40 | 74.13 |
| LlamaGuard3 | 84.07 | 83.96 | 82.78 | 82.81 | 83.45 |
| PG-Qwen-Smol | 86.59 | 85.96 | 84.68 | 85.28 | 86.55 |
| **CREST-LARGE** | **86.06** | **86.08** | **85.65** | **85.33** | **85.55** |

### 低资源语言零样本迁移
**OOD低资源语言表现**：
- CREST-BASE：ID语言平均F1 **82.12**，OOD低资源语言**81.73**（差距仅0.39）
- CREST-LARGE：ID语言平均F1 **84.56**，OOD低资源语言**84.41**（差距仅0.15）

**结论**：零样本迁移性能几乎与训练语言相当，证明聚类引导策略的有效性。

### 代码切换鲁棒性
在**CSRT基准**上：
- CREST-LARGE **全面超越**所有基线
- 所有基线性能显著下降，仅PG-Qwen系列保持相对稳定
- 证明CREST对真实多语言混合场景具有更强泛化能力

### 消融实验结果

#### 簇内迁移分析（RQ1）
在**印度语系簇**（15种语言）内：
- **印地语训练**：平均F1 **85.62**（高资源）
- **卡纳达语训练**：平均F1 **84.84**（中资源）
- **信德语训练**：平均F1 **78.03**（低资源）

**结论**：训练语言资源越高，簇内迁移效果越好。

#### 直接监督 vs 跨语言迁移（RQ2）
- **阿萨姆语**：直接监督84.65 vs 印地语迁移83.91
- **信德语**：直接监督85.59 vs 印地语迁移86.11

**结论**：对极低资源语言，跨语言迁移效果可与直接监督媲美甚至略优。

#### 跨簇迁移分析（RQ3）
- **印度语系簇评估**：印地语训练模型（86.26）>> 中文训练模型（82.68）
- **东亚-东南亚簇评估**：中文训练模型（85.83）>> 印地语训练模型（85.07）

**结论**：同簇迁移显著优于跨簇迁移，但印地语→东亚簇的迁移强于中文→印度语系簇，显示印地语表示更具普适性。

---

## 4. 关键结论和发现

### 主要发现
1. **聚类引导的跨语言迁移是有效的**：通过在13种高资源语言上训练，可实现对100种语言的可靠安全分类，ID与OOD性能差距<0.5%
2. **模型规模与多语言性能正相关**：CREST-LARGE通过更大容量缓解多语言模型的"能力稀释"问题，在复杂推理场景表现更优
3. **脚本相似性影响稳定性**：拉丁脚本语言（如加利西亚语、斯洛文尼亚语）表现更稳定，因XLM-R预训练词汇覆盖更好
4. **高资源语言的选择至关重要**：印地语等语言产生的表示更具跨簇迁移能力，可作为"桥梁语言"
5. **数据效率革命性提升**：相比PolyGuard的1.91M样本，CREST仅用30k样本的翻译版本即实现竞争力性能

### 方法局限性
1. **翻译质量依赖**：机器翻译可能丢失文化细微差别和语气，影响低资源语言的真实危害判断
2. **缺乏显式推理能力**：作为分类模型，未集成LLM的链式推理能力，可能遗漏需要深度理解的复杂安全场景
3. **文化适应性有限**：在Cultural Kaleidoscope基准上表现（56.79-69.42 F1）低于PG-Qwen（75.71），显示对区域文化规范理解不足
4. **模型发布限制**：CREST-LARGE作为商业模型未开源，仅Base版本公开

### 未来工作方向
1. **上下文感知安全建模**：探索基于轻量级多语言LLM的上下文理解能力，提升 nuanced safety judgments
2. **文化特异性适配**：开发区域化安全标准，减少翻译偏差，增强对本地文化规范的敏感性
3. **动态聚类优化**：研究在线聚类方法，适应语言演化和新兴网络用语
4. **边缘部署优化**：进一步压缩模型尺寸，支持在智能手机、IoT设备等极低功耗环境运行
5. **多模态扩展**：将框架扩展至图像、音频等多模态内容的安全审核

---

**核心价值**：CREST首次在0.5B参数规模下实现100种语言的有效安全护栏，为构建包容性、可扩展的全球AI安全基础设施提供了实践验证的技术路径。

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

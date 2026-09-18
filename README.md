# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-09-18 10:06:21 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Block Parallelism For Efficient Distributed Long-Context Diffusion Language Model Training](https://arxiv.org/abs/2609.19242)

**Authors**: Tarun Suresh, Pranshu Chaturvedi, Hangoo Kang, Parth Shroff, Ishan S. Khare, Hermann Kumbong, Azalia Mirhoseini  
**Category**: cs.LG  
**Published**: 2026-09-18  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2609.19242v1  

#### Abstract
Block diffusion language models (BDLMs) combine autoregressive dependencies across blocks with parallel denoising within blocks, but long-context training is constrained by distributed attention communication and activation memory. Conventional context parallelism (CP) shards the combined clean-plus...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Block Parallelism For Efficient Distributed Long-Context Diffusion Language Model Training

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**长上下文扩散语言模型（BDLM）训练中的效率瓶颈**，特别是：
- **分布式注意力通信开销大**：传统 Context Parallelism (CP) 在训练过程中需要在设备间频繁交换干净（clean）和被污染（corrupted）的 Key/Value（K/V）及其梯度，导致大量不必要的跨设备通信。
- **激活内存占用高**：由于每个设备都需存储完整的干净前缀副本，导致**干净前缀的重复存储**，尤其在长上下文场景下内存消耗严重。

### 提出的新方法
提出了两种新的并行策略：
- **Block Parallelism (BP)**：
  - 将每个目标块（target block）的完整计算（前向、损失、反向传播）分配给一个独立的 rank（GPU）。
  - 利用 BDLM 目标函数在不同块之间的可分离性，使每个块的 corrupted K/V 和梯度保持在本地，**消除块间通信**。
- **Context-Sharded Block Parallelism (CSBP)**：
  - 在 BP 的基础上，进一步对**共享的干净序列（clean sequence）进行分片（sharding）**，即在拥有目标块计算的相同 ranks 上应用 CP 来分发干净序列的计算和存储。
  - 这样，只有干净的 K/V 及其梯度需要跨 rank 通信，而 corrupted K/V 完全保留在本地。

### 相比现有方法的优势
- **通信效率更高**：CSBP 移除了 corrupted K/V 的跨设备通信，显著减少了通信量（例如，在 DiffusionGemma 上减少 93.5% 的逻辑注意力流量）。
- **内存更优**：避免了干净前缀的重复存储，将每个 rank 的干净前缀存储从接近 `L` 降低到 `L/P`，实现了高达 `P×` 的内存节省潜力。
- **吞吐量提升显著**：在多种模型和任务上均实现了 1.18× 至 7.59× 的吞吐量加速。
- **保持训练语义**：完全保留了原始 BDLM 的计算和梯度，保证了训练结果的一致性。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Supervised Fine-Tuning (SFT)**：
  - `CoderForge-Preview` 轨迹数据（用于 SWE-bench Verified 评估）。
  - `Terminus-2` 轨迹数据（用于 Terminal-Bench Lite 评估）。
- **AR-to-BDLM Conversion**：使用预训练的自回归模型（如 Qwen3.5-27B, Qwen3.8-27B）进行转换，不依赖特定外部数据集。
- **Speculative Decoding Drafter Training**：基于 `SpecForge` 框架训练 DFlash2 drafter。

### 实验设置和评估指标
- **硬件平台**：
  - 全模型训练：2 节点 × 8 NVIDIA H200 GPU（共 16 张），每张 141 GB HBM3e。
  - DFlash2 训练：1 节点 × 8 NVIDIA H100 GPU（共 8 张），每张 80 GB HBM3。
- **上下文长度**：涵盖 64K 到 1M 的超长上下文。
- **评估指标**：
  - **Tok/s**：每秒处理的 token 数，衡量吞吐量。
  - **MFU (Model FLOPs Utilization)**：模型浮点利用率。
  - **Peak HBM**：GPU 最大峰值显存占用。
  - **Pass Rate**：在 SWE-bench Verified 和 Terminal-Bench Lite 上的通过率，用于下游能力评估。
  - **End-to-end Speedup**：相对于最佳基线的端到端加速比。

### 基线方法对比
- **Baseline**：高度优化的传统训练方法，使用标准的 **Context Parallelism (CP)** 对整个 clean+corrupted 序列按位置分片。
- **对比维度**：在相同的模型、全局 batch size、优化器配置下，仅改变分布式并行策略，比较 CSBP 与最佳非 BP 配置的性能差异。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
#### （1）SFT 和 AR-to-BDLM 转换（16× H200, 256K 上下文）
| 任务 | 模型 | 加速比 (Speedup) | MFU 提升 | Peak HBM 变化 |
|------|------|------------------|----------|---------------|
| BDLM SFT | NemotronDiffusion 3B–14B | **1.18–1.19×** | ↑ ~6% | ≈ 或 ↓ |
| BDLM SFT | DiffusionGemma 26B-A4B | **1.45×** | 12.7% → 18.4% | 118.1 → 113.6 GiB |
| AR-to-BDLM | Qwen3.8-27B | **1.33×** | 20.1% → 26.8% | 120.7 → 111.1 GiB |

> ✅ **结论**：在 256K 上，CSBP 在所有测试模型上均实现 **1.18–1.45× 吞吐提升**，同时匹配或降低峰值显存。

#### （2）扩展至更长上下文（512K）
- **DiffusionGemma 26B-A4B** 在 512K 上达到 **1.61×** 的全模型加速。
- 随着上下文增长，CSBP 的优势持续扩大（图5），证明其在超长上下文下的可扩展性。

#### （3）DFlash2 Speculative Decoder 训练（8× H100）
| 上下文 | 加速比 |
|--------|--------|
| 512K | **2.48×** |
| 1M | **7.59×** |

> 💡 **原因**：传统 CP 会复制整个 block-local decoder 结构，而 CSBP 每个 draft block 仅在一个 rank 上执行，避免了 `P` 倍的冗余计算。

#### （4）下游任务表现（12小时 SFT）
在相同训练时间预算下，CSBP 训练的模型在以下基准上**每个检查点的通过率均更高**：
- **SWE-bench Verified**：最终领先 **1.8 个百分点**。
- **Terminal-Bench Lite**：最终领先 **2.0 个百分点**。

---

### 消融实验结果
#### （1）纯 Block Parallelism (BP) vs. CSBP
- **纯 BP 存在严重内存问题**：
  - NemotronDiffusion 14B 在 128K 时即 OOM。
  - DiffusionGemma 26B-A4B 在 64K 和 128K 均 OOM。
- **CSBP 成功解决此问题**：通过分片干净上下文，使训练在长上下文下变得可行且高效。

#### （2）负载均衡（Dual-end Strategy）
- 使用“首尾配对”策略平衡前后块的计算负载。
- 结果显示，相比连续分配，该策略带来额外 **1.28–1.34×** 的加速，且不增加内存。

#### （3）不同 Block Size 的鲁棒性
- 在 block size 从 128 到 1024 的范围内，CSBP 始终保持稳定优势（DiffusionGemma 上 1.33–1.44×），表明方法对超参数不敏感。

---

## 4. 关键结论和发现

### 主要发现
1. **BDLM 的目标函数天然支持块级并行**：利用其块间损失可分离性，是设计高效并行策略的关键洞察。
2. **CSBP 显著提升训练效率**：通过将 corrupted 计算本地化 + 干净上下文分片，实现了通信和内存的双重优化。
3. **加速效果随上下文增长而增强**：越长的上下文，传统方法通信开销越大，CSBP 的优势越明显。
4. **更高的训练吞吐直接转化为更强的下游能力**：在固定时间内，CSBP 能训练更多步数或更高频次的检查点，从而获得更强的模型。

### 方法的局限性
- **仅适用于训练阶段**：CSBP 依赖于已知的训练目标块，无法用于推理阶段生成未知未来 token。
- **主要针对长上下文场景**：在短上下文下，通信开销占比小，优势可能不明显。
- **需要协调 BP 与 CP 的拓扑**：需合理设计 block 分配与 clean context 分片策略以实现负载均衡。

### 未来工作方向
- 将 BP 思想扩展到其他具有局部-全局结构的模型（如 hierarchical models）。
- 探索 BP 与其他并行范式（如 Pipeline Parallelism）的更深层次融合。
- 设计自动化的并行策略搜索系统，动态选择最优的 BP/CP 组合。
- 将类似思想应用于推理阶段的 speculative decoding 架构优化。

> 🔗 **代码开源**：https://github.com/ScalingIntelligence/Turbo-dLLM

</details>

---

### 2. [DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression](https://arxiv.org/abs/2609.19969)

**Authors**: DeepSeek-AI,  :, Anyi Xu, B. Li, Bangcai Lin, Bing Xue, BingCheng Xian, Bingzheng Xu, Bochao Wu, Bowei Zhang, Boyi Deng, C. C. Yu, Chao Jin, Chaofan Lin, Chen Dong, Chenbing Wang, Chenfan Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chengyuan Zhang, Chenhao Xu, Chenqi Zhao, Chenze Shao, Chuhao Wang, Chuqi Zhang, Damai Dai, Dejian Yang, Deli Chen, Di Huang, Di Wu, Donghao Li, Erhang Li, Eric Fu, F. Zhou, Fangwei Zhou, Fangyun Lin, Fangzhou Yuan, Feiyu Xia, Fucong Dai, Guangbo Hao, Guanglin Li, Guanting Chen, Guoai Cao, Guofan Fan, Guolai Meng, Guowei Li, Haichuan Zhang, Haiyang Ma, Haiyang Shen, Han Li, Han Yu, Han Zhang, Hangyuan Deng, Hanwei Xu, Hanxiang Xu, Hanxun Zhong, Hao Guo, Hao Jiang, Hao Li, Hao Qin, Haodong Wen, Haofen Liang, Haofeng Huang, Haohua Liu, Haoling Zhang, Haoming Luo, Haoran Yang, Haotian Xu, Haotian Yuan, Haoting Huang, Haowen Luo, Haoyang Cai, Haoyu Chen, Haozhe Ji, Hengran Zhang, Hengrui Wang, Hengxu Wu, Honghui Ding, Hongxuan Tang, Huadong Wang, Huanqi Cao, Huazuo Gao, Hui Qu, Hui Zeng, J. Yang, J. H. Jin, J. H. Zhang, J. X. Zou, Jia Yu, Jiahui Zhou, Jiajun Chen, Jialiang Huang, Jialin Zhao, Jiamin Tang, Jian Zhou, Jianan Tong, Jianwen Li, Jiaqi Zhu, Jiarui Wang, Jiasheng Ye, Jiashi Li, Jiaxin Xu, Jiaying Ding, Jibai Lu, Jiewen Hu, Jin Yan, Jincheng Zhai, Jingchang Chen, Jingcheng Hu, Jingli Zhou, Jingsheng Xu, Jingting Xiang, Jingyan Yun, Jingyang Yuan, Jingyuan Cheng, Jinhua Zhu, Jinpeng Wang, Jinyi Chen, Jinyi Hu, Jiping Yu, Jueliang Guo, Junbo Pei, Junbo Sun, Junguang Jiang, Junjie Qiu, Junkang Zhou, Junqi Liu, Junren Li, Junxian Li, Junxiao Song, Junyi Guo, Kai Dong, Kaifeng Chen, Kaige Gao, Kang Guan, Kangdong Yuan, Ke Hong, Ke Xu, Kefan Zhao, Kexin Ji, Kexin Zhang, Kexing Zhou, Kuai Yu, Lan Zhang, Lean Wang, Lecong Zhang, Lei Wang, Letian Gao, Liang Zhao, Liansheng Xu, Lihua Guo, Lingxiao Luo, Lingyue Fu, Litao Deng, Litong Wang, Liyue Zhang, Longhao Chen, Lu Chen, Luotian Huang, Luyao Ma, Luyao Wang, M. S. Di, Max Mei, Menghao Ye, Miao Cui, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingjing Zhang, Mingqi Wei, Mingshu Chen, Mingxing Liu, Mingxu Zhou, Mingyu Xu, Mingyu Yang, Mingze Wang, Muyang Chen, Ni Shentu, Ning Wang, Niufang Ning, Panpan Huang, Peixin Cong, Peiyi Wang, Peiyuan Xin, Pengfei Ren, Pengfei Yan, Pengle Zhang, Qi Kang, Qi Tang, Qiancheng Wang, Qiang Li, Qihao Zhu, Qingyang Li, Qinyu Chen, Qiushi Du, Qizhou Guo, Rongxian Xu, Rui Ding, Rui Hu, Rui Tian, Rui Yu, Ruidong Zhu, Ruifan Xu, Ruihan Yang, Ruihang Xia, Ruijie Lu, Ruilin Geng, Ruipeng Hong, Ruiqi Ge, Ruisong Zhang, Ruize Sun, Ruizhe Pan, Runji Wang, Runqian Chen, Runxin Xu, Ruohong Tian, Ruomeng Shen, Ruoyu Zhang, Ryan X., S. H. Liu, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaofei Cai, Shaoheng Nie, Shaoyuan Chen, Shengding Hu, Shengkai Lin, Shengwen Ran, Shengyu Liu, Shengyuan Jia, Shi Bai, Shi Feng, Shicheng Xu, Shichun Liu, Shiqiang Hu, Shirong Ma, Shiyu Wang, Shiyuan Feng, Shufan Gong, Shuhan Lin, Shuiping Yu, Shunfeng Zhou, Shuo Yang, Shuomeng Wang, Shuting Guo, Shuting Pan, Shuying Yu, Sinuo Cao, Siyi Lin, Sizhe Chen, Songyang Chen, Songyang Zhou, Tao Ni, Tao Yun, Tian Jin, Tian Pei, Tian Ye, Tianle Lin, Tianran Ji, Tianyi Cui, Tianyuan Yue, Tingting Yu, Tongrui Xiong, Wangding Zeng, Wei Liu, Wei Zhang, Weibin Xu, Weihao Zeng, Weilin Zhao, Wen Liu, Wenfeng Liang, Wenjie Pang, Wenjing Luo, Wenjing Yao, Wenjun Gao, Wenkai Shao, Wenkai Yang, Wenli Zhang, Wenlu Wang, Wenlve Huang, Wenqian Yan, Wentao Zhang, Xi Gao, Xiang He, Xiang Li, Xiangli Li, Xiangwen Wang, Xiangying Zhang, Xiankui Wei, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojian Qu, Xiaokang Chen, Xiaokang Zhang, Xiaotao Nie, Xiaoyao Zou, Xiaoyuan Li, Xicheng Guo, Xieting Chu, Xin Cheng, Xin Liu, Xin Xie, Xinbo Xu, Xingchao Liu, Xingchen Liu, Xingkai Yu, Xingyou Li, Xintong Yao, Xinyang Chen, Xinyong Jiang, Xinyu Yang, Xinyu Yang, Xu Chen, Xuanyu Wang, Xubei Zhong, Xuecheng Su, Xuejie Liu, Xuheng Lin, Xujie Fan, Xuncheng Zhao, Xuwei Fu, Y. C. Yan, Y. H. Jiang, Y. T. Wu, Y. W. M., Y. Z. Wang, Yafei Gao, Yang Yang, Yang Zhang, Yanru Ma, Yanwen Huang, Yao Li, Yao Li, Yao Meng, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yaoyang Ye, Yehang Yin, Yexinrui Wu, Yi Qian, Yi Tao, Yi Yu, Yichao Zhang, Yichen Jiang, Yicheng Wang, Yifan Ding, Yifan Shi, Yifeng Peng, Yifeng Zhai, Yijia Wu, Yiliang Xiong, Yilun Wang, Ying He, Ying Zhou, Yingjia Luo, Yinmin Zhong, Yiping Wang, Yisong Wang, Yixiang Zhang, Yixiao Chen, Yixuan Tan, Yixuan Wei, Yiyang Ma, Yiyao Yang, Yiyuan Liu, Yizai Cai, Yizhen Wei, Yizhi Wang, Yonglun Yang, Yongqi Zhuo, Yongqiang Guo, Yongtong Wu, Yu Wu, Yu Zhang, Yuan Bian, Yuan Cheng, Yuan Ou, Yuan Sun, Yuanfan Xu, Yuanhang Sun, Yuanhao Li, Yuchen Liu, Yuchen Yao, Yudong Han, Yuduan Wang, Yuhan Wu, Yuhao Meng, Yuheng Zou, YuKun Li, Yunchuan Wang, Yunfan Xiao, Yunfan Xiong, Yupeng Chen, Yuqian Cao, Yuqian Wang, Yuqing Chen, Yushun Zhang, Yutong Lin, Yuwei Xiao, Yuxian Gu, Yuxiang Chen, Yuxiang Huang, Yuxiang Luo, Yuxiang You, Yuxin Chen, Yuxin Xiang, Yuxuan Liu, Yuxuan Zhou, Yuyang Zhou, Yuzhe Guo, Yuzhen Huang, Yuzhuo Bai, Z. Y. Z., Zanlin Ni, Zehao Wang, Zehua Zhao, Zehui Ren, Zejun Zhao, Zhangli Sha, Zhanying Wang, Zhaochen Zhang, Zhaoshuai Du, Zhe Fu, Zhean Xu, Zhenda Xie, Zheng Liu, Zhengyan Zhang, Zhenhua Dong, Zhewen Hao, Zhibang Wang, Zhibin Gou, Zhicheng Ma, Zhihao Li, Zhihong Shao, Zhihuan Huang, Zhijie Li, Zhirui Lu, Zhixian Huang, Zhixuan Chen, Zhixuan Chen, Zhixuan Pan, Zhiyu Wu, Zhizhou Ren, Zhu He, Zhuoshu Li, Zhuping Zhang, Zian Xu, Zihao Wang, Zihui Gu, Zijia Zhu, Zili Zhang, Zilin Li, Zilong Hou, Zilong Lyu, Ziqiao Wang, Ziwei Xie, Ziya Zhang, Ziyi Gao, Zizheng Pan, Zonglin Li, Zongqing Yao, Zui Chen, Zuofan Wu, Chenchen Ling, Chengyu Hou, Chong Chen, D. Li, Di Qi, Dongjie Ji, Fang Wei, Fanyi Xia, Fei Xie, Feiyi Tan, Hailong Guo, Haiyan Zhai, Hui Zhou, Huihui Tan, Huijie Li, Jia Luo, Jia Song, Jialu Cai, Jian Liang, Jiangting Zhou, Jiaqi Gao, Jiayi Shao, Jie Chen, Jieyu Yang, Jin Chen, Jingde Zhang, Jingzi Zhou, Jinqian Wang, Jinyang Liu, JinZhao Sun, Junhua Ling, Junmin Zheng, Kaicheng Yang, Ke Xu, Le Su, Leyi Xia, Liangfeng Ding, Lin Zhuo, Linwang Ma, Linyan Zhu, Liyu Cai, Luqi Yao, M. K. Zhang, Meng Li, Miao Lin, Miaojun Wang, Min Zhang, Mingming Li, Mingming Wang, Mingze Yin, Minmin Han, Nan Cao, Ning Wang, Ningxin Ma, Panpan Wang, Peihan Lin, Peng Sun, Peng Zhang, Qian Ying, Qiang Xiang, Qiao Wang, Qingmiao Mao, Qiwei Jiang, Rongli Jin, Ruyi Chen, Sha Tao, Shangmian Sun, Shaoqing Wu, Shichao Zou, Si Lei, Tianyang Zhang, Tianyu Sun, Tingting Yin, W. L. Xiao, Wei An, Wei Li, Wei Wang, Weiwei Lin, Wenqing Hou, X. Lin, Xiangfei Meng, Xianzhu Huang, Xiao Peng, Xiaoqian Li, Xiaoting Zhang, Xiaowen Sun, Xiaoxiang Wang, Xiaoyu Ye, Xinrou Zhang, Xinyu Zhang, Xue Cao, Xueyin Chen, Yanan Zhou, Yanhong Xu, Yao Xia, Yao Xu, Yi Shao, Yihong Zhang, Yiling Ma, Ying Tang, Yining Lou, Yiru Chen, Yishi Piao, Yixuan Chen, Yong Xiong, Yuchen Xuan, Yuehan Yang, Yuer Xu, Yukun Zha, Yunxian Ma, Yuping Lin, Yuting Yan, Yutong Xie, Yuwen Sheng, Yuxuan Zhu, Zekai Zhang, Zhe Ju, Zhenzhen Lin, Zheren Gao, Zheyang Sun, Zhigang Yan, Zhongyu Wu, Zi Wang, Zihua Qu, Ziling Yan, Ziyi Wan  
**Category**: cs.CL  
**Published**: 2026-09-18  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2609.19969v1  

#### Abstract
The widespread adoption of long-horizon agents has made model workloads increasingly input-heavy. Although prior work has substantially reduced the cost of long-context computation, prefill remains computationally expensive, and large KV caches continue to strain HBM and SSD capacity and data-transf...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression

---

## 1. 论文的主要贡献和创新点

### 解决的问题
随着 **long-horizon agents** 的广泛应用，模型在处理超长上下文时面临严重的计算、存储和带宽瓶颈。特别是 **KV Cache** 在推理阶段占用大量 HBM 和 SSD 容量，并且其传输开销显著影响部署成本。尽管已有工作优化了长序列计算（如稀疏注意力），但 **prefill 阶段的计算成本** 和 **KV Cache 的持久化存储需求** 仍是主要瓶颈。

### 提出的新方法与创新思路
为解决上述问题，DeepSeek 团队提出 **DeepSeek-V4.1-Flash**，一个支持百万级上下文的多模态 MoE 模型，通过以下三大协同优化实现 KV Cache 极致压缩：

#### （1）**Causal Encoder-Decoder (CED) 架构**
- 将 Transformer 分为因果编码器（前半部分）和解码器（后半部分）。
- 解码器的全局 KV Cache 由编码器最后一层的隐藏状态投影而来，避免重复计算。
- **优势**：Prefill 阶段仅需运行前半层，计算复杂度从 $O(NL)$ 降至 $O(NL/2)$，大幅降低输入密集型任务的成本。

#### （2）**Compressed Sparse Attention 2 (CSA2)**
- 在全局注意力中引入跨层 KV 共享机制，包含三种静态模式：
  - **Full Mode**：生成新的主 KV 和索引。
  - **Reindex Mode**：复用主 KV 和索引器 K，重新打分选择 Top-K。
  - **Reuse Mode**：直接复用主 KV 和 Top-K 索引。
- 引入 **Hierarchical Sparse Indexer**：首层 Full Mode 层构建候选池，后续层仅在此池内搜索，将索引成本从线性降为常数。
- **优势**：显著减少 KV 存储和索引计算，提升效率。

#### （3）**FP4KV 缓存 + SWA Bounded Replay**
- **FP4 Main KV Cache**：采用量化感知训练（QAT）将主 KV Cache 压缩至 FP4 精度，几乎减半存储。
- **SWA Bounded Replay**：放弃持久化存储滑动窗口 KV（SWA KV），在缺失时只需重放最近 `n_win` 个 token 即可近似重建。
- **优势**：持久化 KV Cache 脚印减少至 DeepSeek-V4-Flash 的 **1/8**，极大缓解 SSD 压力。

### 相比现有方法的优势
| 维度 | DeepSeek-V4.1-Flash | DeepSeek-V4-Flash |
|------|---------------------|--------------------|
| 每 token 全局 KV Cache 大小 | **890 字节**（HBM） | ~3,514 字节 |
| 持久化 KV Cache 脚印 | **~1/8** | 基准 |
| Prefill 参数激活 | **8B/token** | 13B/token |
| Decode 参数激活 | **16B/token** | 13B/token |
| 上下文长度支持 | **1M tokens** | 1M tokens |
| 性能表现 | **全面优于或持平** | 基线 |

> ✅ **核心优势**：以更小的 KV Cache 脚印和更低的 prefill 成本，实现了更强的整体性能。

---

## 2. 核心实验方法和设置

### 使用的数据集
#### 预训练数据（共 45T tokens）
- **文本数据**：高质量网页、代码仓库（GitHub）、学术文献等，过滤低质量生成内容。
- **多模态数据**：
  - 图文对（image-text pairs）
  - 交错图文数据（interleaved web pages/PDFs）
  - 领域特定数据（如图表、OCR、图像-代码对）

#### 后训练（Post-Training）数据
- 自动化合成的 **agent task triplets**（问题、环境、验证系统）
- 内部员工真实工作流反馈重构的任务
- GitHub 高星项目构建的编码环境
- 多轮强化学习（RL）轨迹，覆盖软件工程、网络安全、自动化办公等场景

### 实验设置与评估指标

| 类别 | 指标 | 主要基准 |
|------|------|----------|
| **世界知识** | EM | AGIEval, MMLU-Pro, C-Eval, SuperGPQA |
| **语言理解与推理** | EM/F1 | BBH, BBEH, DROP, HellaSwag |
| **编程与数学** | Pass@1/EM | HumanEval, GSM8K, MATH, BigCodeBench |
| **长上下文能力** | EM | LongBench-V2 |
| **多模态能力** | EM/Judge | MMMU-Pro, DocVQA, CVBench, RefCOCO |
| **代理能力（Agentic）** | Pass@1/Resolved | Terminal-Bench 2.1/3.0/4.0, DeepSWE v1.1, AutomationBench |
| **安全能力** | Pass@1 | SEC-Bench Pro, CyberGym, ExploitGym |

### 基线方法对比
- **开源模型**：Kimi-K3, GLM-5.3, DeepSeek-V4-Flash, DeepSeek-V4-Pro
- **闭源模型**：GPT-5.6 Sol, Opus-5, Claude Code
- 所有评估均在统一框架下进行，控制温度（1.0）、top-p（0.95）等参数一致。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 3）

| 基准 | DeepSeek-V4.1-Flash (Max Effort) | 最佳竞品 |
|------|-------------------------------|--------|
| **GPQA Diamond (Pass@1)** | 90.9% | 94.1% (Opus-5) |
| **Codeforces (Rating)** | **3471** | 3348 (DS-V4-Pro) |
| **MathArena Apex (Pass@1)** | **65.6%** | 65.6% (Kimi-K3) |
| **Terminal-Bench 2.1 (Pass@1)** | **90.6%** | 89.1% (Opus-5) |
| **DeepSWE v1.1 (Resolved)** | **74.2%** | 74.0% (Opus-5) |
| **CyberGym (Pass@1)** | **88.1%** | 84.5% (Opus-5) |
| **Agents' Last Exam (Pass@1)** | **31.8%** | 28.6% (Opus-5) |

> 🔥 **亮点**：在多个核心代理任务上超越闭源前沿模型，尤其在 **DeepSWE v1.1** 和 **CyberGym** 上达到 SOTA。

### 与基线方法的对比结果
- **KV Cache 压缩效果**：
  - 全局 KV Cache per token：**890 bytes** → 较 V4-Flash 减少 **~4倍**
  - 持久化 KV Cache：减少 **~8倍**
- **推理效率**：
  - 单 token Decode FLOPs 几乎不随上下文增长而增加（见 Figure 2）
  - 支持 1M 上下文下仍保持高吞吐与低延迟
- **性能提升**：
  - 在相同上下文长度下，性能全面优于 DeepSeek-V4-Flash 和 DeepSeek-V4-Pro
  - 多模态理解能力优于 Kimi-K3，尤其在专业图表分析方面

### 消融实验结果（隐含于设计分析中）
虽然未提供显式消融表，但从架构演进可推断各组件贡献：

| 组件 | 贡献估计 |
|------|--------|
| **CED 架构** | Prefill 计算减少约 50%，显著降低成本 |
| **CSA2 + Cross-layer Reuse** | KV 存储减少 3–4 倍，索引计算大幅下降 |
| **FP4KV** | 主 KV Cache 存储再减半 |
| **SWA Bounded Replay** | 消除 SWA KV 持久化开销，节省约 50% 持久化容量 |
| **Head-wise Muon + Sinkhorn Balancing** | 提升训练稳定性与收敛速度 |

> 💡 综合效应使 KV 脚印总压缩达 **~8倍**，同时性能反超更大模型。

---

## 4. 关键结论和发现

### 主要发现
1. **KV Cache 压缩是降低长上下文部署成本的关键路径**  
   通过 **架构设计（CED/CSA2）+ 精度压缩（FP4）+ 部署策略（Bounded Replay）** 的联合优化，可在不牺牲性能的前提下极致压缩 KV 脚印。

2. **DeepSeek-V4.1-Flash 实现“更小更强”**
   - 激活参数更少（prefill 8B vs 13B），但性能全面领先。
   - KV Cache 更小，却在 **代理、编程、安全** 等复杂任务上超越闭源模型。

3. **可控推理努力（Controllable Reasoning Effort）有效**
   - 用户可通过调节 `effort` 参数（1–100）灵活控制输出长度与准确率。
   - 在 API 中暴露 **low (50), high (75), max (100)** 三档，适配不同延迟与预算需求。
   - 努力值从 25 提升到 100，平均 Pass@1 提升 9.2%，代价约为 2.5× 输出 token。

4. **多智能体协作优于单智能体**
   - 在 ProgramBench 和 FrontierSWE v2 上，multi-agent 配置在所有时间限制下均优于 single-agent。
   - 表明任务分解与并行执行是应对复杂长程任务的有效范式。

### 方法的局限性
1. **边界情况下的鲁棒性尚未完全验证**
   - CSA2 的索引复用和 Bounded Replay 的近似重建可能在极端输入下导致性能退化。
   - 当前测试集无法覆盖所有边缘案例。

2. **对最困难任务仍有差距**
   - 在如 Terminal-Bench 4.0 这类需要专家领域知识的任务上，仍落后于顶级闭源模型（如 GPT-6 Astra）。
   - 表明模型在“深度专业知识”上的积累仍需加强。

3. **评估饱和问题**
   - 多数标准 benchmark 已接近上限，难以区分顶尖模型的真实差距。
   - 需要更难、更具挑战性的新评测体系。

### 未来工作方向
1. **持续扩展 stress-testing 与评估体系**
   - 加强对稀疏检索、长上下文恢复边界的测试。
   - 开发更具挑战性的 reasoning 和 agent benchmarks。

2. **协调扩展（Coordinated Scaling）**
   - 推动 **data、model capacity、RL** 的联合扩展，突破当前智能上限。
   - 利用 DeepSeek-V4.1-Flash 作为新起点，探索更大规模的 post-training。

3. **模型-框架协同设计（Model-Harness Co-design）**
   - 将模型与 agent framework 一起优化，形成闭环进化系统。
   - 提升工具调用、记忆管理、多步规划等能力。

4. **进一步降低部署门槛**
   - 结合 KV 压缩成果，推动长上下文 agent 在中小企业和个人用户的普及。
   - 探索边缘设备上的轻量化部署方案。

---

> 📌 **总结**：DeepSeek-V4.1-Flash 不仅是一个高性能大模型，更是 **面向实际部署的系统级创新**。它证明了通过软硬协同设计，可以在显著降低成本的同时提升模型能力，为大规模部署长上下文智能代理铺平道路。

</details>

---

### 3. [To Copy or Not to Copy: Controlling Speculative Decoding via Intrinsic Model Signals](https://arxiv.org/abs/2609.20186)

**Authors**: Roy Eisenstadt, Ido Cohen, Edo Cohen-Karlik, Lior Wolf, Itamar Zimerman  
**Category**: cs.CL  
**Published**: 2026-09-18  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2609.20186v1  

#### Abstract
Speculative Decoding (SD) has significantly accelerated Large Language Model (LLM) inference, yet existing approaches face a fundamental tradeoff between two drafting strategies: neural drafting and context-based copying. Neural drafts (e.g., EAGLE3) provide robust performance across diverse text se...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*To Copy or Not to Copy: Controlling Speculative Decoding via Intrinsic Model Signals*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **Speculative Decoding (SD)** 方法在“神经草案”（neural drafting）和“上下文复制”（context-based copying）之间面临根本性的权衡：
- **神经草案**（如 EAGLE3）：鲁棒性强，适用于多样文本，但加速有限。
- **复制草案**（如 CopySpec）：在重复性强的场景中速度极快，但容易因表面 n-gram 重叠而误触发“意外重复”（accidental repetitions），导致大量无效推测，反而降低吞吐量。

核心问题是：**如何准确判断模型是否处于“有意复制”的状态？**

### 提出的新方法：SwitchSD
提出 **SwitchSD** —— 一种自适应的 SD 框架，通过分析目标模型内部表示（internal representations）来动态切换草案策略。

#### 创新思路：
- 将“复制”视为一种由模型内部信号控制的**潜在解码模式**（latent decoding regime），而非简单的表面文本匹配。
- 引入一个轻量级的 **copy-intent probe**，训练其从 LLM 的隐藏层中检测“复制意图”。
- 只有当 probe 高置信度地识别出 copy-intent 时，才启用上下文复制；否则回退到神经草案（如 EAGLE3 或 SPS）。

### 相比现有方法的优势
| 方法类型 | 代表 | 缺陷 | SwitchSD 如何改进 |
|--------|------|------|------------------|
| **Heuristic-based** | CopySpec | 依赖前缀匹配，无法区分“真实复制”与“偶然重复” | 使用内部信号过滤噪声，避免误触发 |
| **Statistical-adaptive** | BanditSpec | 黑箱探索机制，需频繁试错，效率低 | 白盒信号预测，无需探索，决策更精准 |
| **Neural-only** | EAGLE3 | 统一策略，未利用上下文中的长重复机会 | 动态调度，结合两者优势 |

**核心优势**：将“复制”从一种不可靠的启发式技巧转变为一种高精度、模型感知的解码范式。

---

## 2. 核心实验方法和设置

### 数据集
在三个典型任务上进行评估：
- **HumanEval**：代码生成（强结构化、高重复）
- **Math500**：数学推理（混合逻辑与重复）
- **CNN/DailyMail**：摘要生成（弱重复、开放生成）

### 实验设置
- **模型家族**：
  - `LLaMA-3.1-8B-Instruct`, `LLaMA-3.3-70B-Instruct`
  - `Qwen3-8B`（含 8-bit 推理）
- **草案模型**：
  - 使用 `EAGLE3` 或小型 `Qwen3-0.6B` 作为神经草案
- **评估指标**：
  - **Tok/s**：每秒生成 token 数
  - **Speedup**：相对于 vanilla autoregressive 解码的速度提升倍数
  - **Acceptance length**：每次推测平均接受的 token 数

### 基线方法对比
| 类型 | 方法 | 描述 |
|------|------|------|
| Heuristic | PLD, CopySpec | 基于 n-gram 匹配或前缀查找 |
| Neural-only | EAGLE3 | 当前最先进的神经草案方法 |
| Statistical | BanditSpec | 多臂老虎机策略选择，基于吞吐反馈 |
| Ours | **SwitchSD** | 基于内部信号的白盒控制 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 和 Table 2）

| 方法 | HumanEval (Tok/s) | Speedup | Math500 (Tok/s) | Speedup | CNN/DM (Tok/s) | Speedup |
|------|-------------------|---------|------------------|---------|----------------|---------|
| **Vanilla** | 38.3 | 1.00× | 11.7 | 1.00× | 26.0 | 1.00× |
| **EAGLE3** | 87.5 | 2.28× | 88.7 | 2.29× | 77.0 | 2.01× |
| **SwitchSD** | **98.6** | **2.58×** | **88.7** | **2.29×** | **77.0** | **2.01×** |

> ✅ **最高吞吐量**：SwitchSD 在所有配置下均达到最高速度。
>
> ✅ **相对提升**：相比 EAGLE3 最多提升 **15%** 吞吐量。

### 与基线方法的对比结果

| 对比项 | 结果 |
|-------|------|
| **vs. CopySpec** | 触发复制频率更低（如 7.2% vs 14.3%），但平均接受长度更长（4.37 vs 2.08），实现“少而精” |
| **vs. BanditSpec** | 避免探索惩罚，在 Math500 上 BanditSpec 几乎崩溃（仅 0.99×），而 SwitchSD 达到 1.48× |
| **vs. PLD/SAMD/Lookahead** | 所有非参数复制方法均被超越，尤其在长序列接受能力上显著领先 |

### 消融实验结果

#### （1）训练数据的影响（Figure 4）
| 训练数据 | F1 Score | 说明 |
|--------|----------|------|
| ConstructedCopy（随机重复） | 0.65 | 性能差，表明不能仅靠结构 |
| WikiText-103（自然语言） | 0.81 | 有所提升，但仍不足 |
| **CopyDiversity（本文构造）** | **0.87** | 明显最优，证明需要语义对齐的多样化数据 |

#### （2）表示位置的选择（Figure 5）
- **最佳位置**：中间层（如 Llama-3.1-8B 的第14层）
- **子层选择**：**Attention 子层后** > MLP 子层后  
  → 支持“复制意图由 induction head 控制”的假设

#### （3）阈值与可分性（Figure 6）
- **AUC > 0.99**：表明 copy-intent 在隐藏空间中几乎线性可分
- **F1 最优阈值 ≈ 0.4**：可在精度与召回间取得平衡

---

## 4. 关键结论和发现

### 主要发现
1. **“复制”是一种可探测的潜在模式**  
   LLM 内部存在明确的“复制意图”信号，可通过轻量 probe 高精度识别（AUC > 0.99）。

2. **高收益来自“精准触发”，而非“高频触发”**  
   SwitchSD 不是更多地复制，而是**只在真正有利时才复制**，从而避免“意外重复”带来的推测税。

3. **信号感知优于统计反馈**  
   相比 BanditSpec 等黑箱优化器，SwitchSD 利用白盒信号提前决策，避免探索成本，效率更高。

4. **方法具有架构无关性**  
   无论使用 EAGLE3 还是 SPS 作为神经草案，SwitchSD 均能带来增益，说明其作为“编排层”（orchestration layer）的通用性。

5. **支持更精细的系统调优**  
   分离“复制”与“生成”路径后，可为不同路径分别优化 `lookahead` 参数（见 Table 3），提升资源利用率。

### 局限性
- **依赖高质量的 probe 训练数据**：当前使用合成数据（CopyDiversity），未来可能受限于真实场景泛化。
- **引入额外计算开销**：虽然 probe 极轻量，但在极端低延迟场景仍需考虑。
- **假设复制行为是二元的**：目前建模为“复制 vs 生成”，未来可扩展为多模态（如部分复制、模板填充等）。

### 未来工作方向
1. **多模态编排器**：将二分类 probe 扩展为预测最优 `lookahead` 长度或选择多个专用草案模型。
2. **深入解释 copy-circuits**：结合 mechanistic interpretability 技术，进一步理解 induction head 如何编码复制意图。
3. **端到端联合训练**：将 probe 与草案模型共同优化，以更好捕捉任务特定信号。
4. **应用于其他高效推理技术**：如 early exiting、token merging 等，构建统一的模型感知推理框架。

---

> 🔚 **总结一句话**：  
> SwitchSD 成功将“复制”从一种脆弱的启发式操作，转变为一种由模型内在信号驱动的、高精度的解码策略，实现了 **“less is more”** 的加速哲学。

</details>

---

### 4. [Syndrome Decoding for Silent Data Corruption in Quantized Integer GPU Arithmetic](https://arxiv.org/abs/2609.19743)

**Authors**: Pranav Napolean, Vikas Srivastava, Napolean Periathambi  
**Category**: cs.DC  
**Published**: 2026-09-18  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2609.19743v1  

#### Abstract
Quantized neural network inference runs integer matrix multiplications on GPU tensor cores, and the INT32 accumulators inside those cores have neither parity nor ECC. A transient fault in this datapath returns a valid but wrong integer and raises no interrupt. Checksum based Algorithm Based Fault To...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Syndrome Decoding for Silent Data Corruption in Quantized Integer GPU Arithmetic**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
- **Silent Data Corruption (SDC)** 在量化整数 GPU 推理中是一个严重问题，尤其是在基于 INT8 的 LLM 推理中。
- GPU tensor cores 执行 `mma.sync.aligned.m16n8k32` 指令时，INT32 accumulator 中没有 ECC 或 parity 保护，单粒子翻转（SEU）会导致输出为“合法但错误”的整数，且不会触发中断。
- 传统 **Algorithm-Based Fault Tolerance (ABFT)** 方法如 checksum 仅能检测错误，无法定位或修复多个并发错误，且存在**确定性盲点**（如矩形误差模式、Vandermonde kernel 模式），导致漏检。

### **提出的新方法：SProbe**
- **SProbe** 是一个**后置验证核（trailing verification kernel）**，运行在独立 CUDA stream 上，不修改原始 GEMM 内核。
- 包含两个阶段：
  1. **Fast Path: Freivalds Gate**
     - 使用三个随机点在 61-bit 质数域上进行 Freivalds 投影，验证 $ A(Bx) = Cx $。
     - 错误未被检测的概率上限为 $((N-1)/p)^3$，在 $N=16384$ 时低于 $2^{-141}$。
  2. **Slow Path: Syndrome Decoding**
     - 若 gate 触发，则计算每行的 power sum syndromes（基于位置加权）。
     - 使用 **Reed-Solomon 解码链**（Berlekamp-Massey + Chien Search + Forney）恢复最多 **4 个并发错误/行** 的精确列位置和修正值。
     - 利用三质数 **Residue Number System (RNS)** 实现精确整数恢复，并通过跨域一致性检查提升可靠性。

### **相比现有方法的优势**
| 特性 | TR-ABFT | Grid Code | SProbe |
|------|--------|----------|--------|
| 检测能力 | ✅ 单错误/块 | ✅ 多错误 | ✅ 高概率无遗漏 |
| 定位能力 | ❌ | ✅ 局部定位 | ✅ 精确到元素 |
| 修复能力 | ❌ | ✅ 最多 2×2 子矩阵 | ✅ 最多 4 错误/行 |
| 盲点抗性 | ❌ 矩形模式、模倍数 | ❌ Vandermonde kernel | ✅ 全部可检测并修复 |
| 不修改 GEMM | ✅ | ❌ 需编码输入 | ✅ |
| 支持诊断与遥测 | ❌ | ❌ | ✅ 输出设备 UUID、层、行列、翻转位等 |

> ✅ **核心创新**：首次将 **syndrome decoding + RNS + GPU kernel design** 结合，实现对量化 INT8 GEMM 的**精确错误定位与修复**，超越传统 checksum 和 grid code 的能力边界。

---

## **2. 核心实验方法和设置**

### **测试平台**
- **硬件**：NVIDIA H100 80GB HBM3（Hopper 架构）
- **软件栈**：CUDA 13.0, PyTorch 2.13.0, Triton 3.7.1, CuPy 14.1.1
- **驱动版本**：580.126.20

### **数据集与模型**
- **主任务**：OpenBioLLM（8B 参数 Llama 3 医疗 LLM）
- **评估任务**：MedQA（USMLE 风格医学问答），共 60 个问题，其中 43 个基础正确
- **注入方式**：
  - **软件注入**：直接修改 INT32 accumulator 值（模拟 bit flip）
  - **硬件注入**：自研 NVBit 工具，在 Hopper tensor core 指令级注入故障（验证真实性）

### **评估指标**
| 指标 | 定义 |
|------|------|
| **SDC rate** | 正确答案变为错误的比例 |
| **Generation corruption rate** | 整体生成文本发生变化的比例 |
| **Detection rate** | 成功检测出注入错误的比例 |
| **Correction rate** | 成功精确定位并修复错误的比例 |
| **Latency overhead** | SProbe 验证开销占 cuBLASLt GEMM 时间比例 |
| **Throughput cost** | 端到端推理吞吐下降百分比 |

### **基线方法**
1. **TR-ABFT** [5]
   - 基于模 $2^{39}$ 的 tile-level checksum，仅检测，无法定位。
2. **Grid Code** [7]
   - 每轴添加 unweighted + index-weighted parity，可纠正最多 2 行×2 列内的错误。

> 所有基线均由作者重新实现并在相同环境下测试，确保公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **检测与修复能力（Table IV）**
| Fault Class | TR-ABFT Det. | Grid Code Corr. | SProbe Det./Corr. | SProbe Action |
|------------|---------------|------------------|--------------------|----------------|
| Single bit | 100% | 100% | 100% | Repair |
| Burst (k=2) | 93.8% | 100% | 100% | Repair |
| Burst (k=4) | 98.8% | 0% | 100% | Repair |
| Burst (k=8) | 100% | 0% | 100% | Recompute |
| Modulo multiples | 0% | 0% | 100% | Repair |
| Rectangle pattern | 0% | 100% | 100% | Repair |
| Vandermonde (k=3) | 0% | 0% | 100% | Repair |

> 🔍 **说明**：
> - TR-ABFT 对所有构造性盲点（modulo multiples, rectangle, Vandermonde）完全失效。
> - Grid Code 可修复 rectangle，但无法处理 Vandermonde kernel 模式（需 >2 错误）。
> - **SProbe 在所有 560 次注入中均成功检测，且在容量内全部精确修复**。

#### ✅ **语义影响（Table III）**
| Bit Range | SDC Rate | Generation Corruption |
|----------|---------|------------------------|
| 0–23     | 0.0%    | 0.0%                   |
| 24–31    | 5.8%    | **84.9%**              |

> 💡 高位翻转虽不一定改变最终答案，但会严重破坏生成文本流，强调必须恢复**完整 INT32 accumulator** 而非仅标记异常。

#### ✅ **性能开销（Table V & VI）**
| Matrix Size (N) | SProbe Gate Overhead (% of cuBLASLt GEMM) |
|------------------|-------------------------------------------|
| 1024             | 303%                                      |
| 4096             | 93%                                       |
| 8192             | 93% → 实际为 1.07/1.15 ≈ 93%               |
| 16384            | **49%**                                   |
| 65536            | **11%**                                   |

> ⚠️ 开销随规模增大而降低（GEMM $O(N^3)$ vs Gate $O(N^2)$）。

#### ✅ **恢复策略对比（Table VI）**
| Configuration | Recompute Time (ms) | In-place Recovery Time (ms) | Ratio |
|--------------|----------------------|-------------------------------|-------|
| N=16384 (square) | 13.33 | 57.76 | 4.33× slower |
| N=262144 (decode shape) | 7.66 | 173.48 | **22.66× slower** |

> 📉 **结论**：**recomputation 比 in-place recovery 更快**，因为 syndrome generation 占 recovery 总时间的 **57.9%**（Table VII），而解码本身仅占 0.3%。

#### ✅ **端到端部署效果**
- 在 OpenBioLLM 上启用 SProbe 后：
  - 无故障时：吞吐从 11.39 → **7.92 tokens/s**（**-30.4%**）
  - 注入 30 次故障：**全部被修复，输出与参考完全一致**
  - 无 false positive 或 miscorrection

---

## **4. 关键结论和发现**

### **主要发现**
1. **SDC 对 LLM 生成质量有显著语义影响**，尤其是高位 bit flip，即使答案不变，文本流也会被严重污染。
2. **传统 checksum ABFT 存在致命盲点**：
   - 矩形误差模式（+e/-e 对角抵消）
   - Vandermonde kernel 模式（3 个 signed power-of-two error 可使 weighted parity 失效）
3. **SProbe 实现了前所未有的错误处理能力**：
   - 检测概率极高（< $2^{-141}$ 漏检率）
   - 可精确定位并修复最多 4 个并发错误/行
   - 支持设备级 telemetry（UUID、row/column、magnitude）
4. **实际恢复中，诊断成本远高于修复成本**：
   - syndrome generation 占主导
   - recomputation 反而更快，适合只关心正确性的场景
5. **SProbe 可无缝集成到生产流程**，无需修改 vendor GEMM（如 cuBLASLt）

### **局限性**
- **Column 0 错误无法修复**：因 locator 为 0，无逆元，只能 fallback 到 recomputation。
- **不支持浮点格式**（FP16/BF16）：依赖 exact integer arithmetic，舍入破坏 syndrome。
- **内存索引溢出风险**：32-bit flat index 在 $N \geq 65536$ 时溢出，需升级至 64-bit。
- **端到端吞吐损失约 30%**：主要来自 host-side synchronization，可通过 batching 优化。
- **硬件注入仅在 N=512 验证**：更大尺寸仍依赖软件注入假设。

### **未来工作方向**
1. **扩展到 INT4 和更宽 accumulator**（如 INT64）：增加更多质数字段。
2. **优化 syndrome generation**：探索低秩近似或稀疏投影以降低 $O(KNS_{\text{max}})$ 成本。
3. **支持在线学习式调度**：根据历史 fault rate 动态选择是否启用 decoding。
4. **构建 fault prediction system**：利用 telemetry 数据预测 GPU degradation。
5. **集成到分布式训练框架**：实现跨节点 SDC 协同诊断。

---

> ✅ **总体评价**：  
> SProbe 是首个将 **coding theory**（Reed-Solomon + RNS）与 **GPU system design** 深度结合的 SDC 防护方案，不仅解决了检测盲点问题，还提供了宝贵的**故障诊断能力**，为 AI 集群的可靠运维提供了新范式。尽管恢复速度不如重算，但其**精准定位能力**是 checksum 和纯重算无法替代的核心价值。

</details>

---

### 5. [P-GADMM: Parallel Group-Based ADMM for Asynchronous Optimization in Heterogeneous Edge Networks](https://arxiv.org/abs/2609.20006)

**Authors**: Gaiguo Wei, Qingying Zhang, Heqiang Wang, Yu Zhang, Xiaoxiong Zhong  
**Category**: cs.DC  
**Published**: 2026-09-18  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2609.20006v1  

#### Abstract
The Alternating Direction Method of Multipliers (ADMM) is widely used for distributed optimization, but its synchronous implementation can suffer from efficiency loss in heterogeneous edge networks, where fast clients or groups need to wait for slower ones before global updates can be completed. Exi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：P-GADMM: Parallel Group-Based ADMM for Asynchronous Optimization in Heterogeneous Edge Networks

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在异构边缘网络（heterogeneous edge networks）中，传统的同步 ADMM 存在严重的“straggler bottleneck”问题：由于客户端在计算能力、数据规模和通信条件上差异显著，全局更新必须等待最慢的客户端完成，导致训练效率低下。此外，现有的分组式 ADMM（如 GADMM）虽然减少了通信开销，但其分组策略通常基于拓扑或数据相似性，**未显式考虑计算异质性**，因此组内仍可能存在速度不匹配的问题。

### 🚀 提出的新方法：P-GADMM
本文提出 **Parallel Group-Based ADMM (P-GADMM)**，一种面向异构边缘网络的并行分组 ADMM 框架，其核心创新包括：

- **计算感知分组策略（Computation-aware Grouping）**  
  客户端根据其计算能力 $C_i$ 和本地数据大小 $|D_i|$ 被划分为若干组，使得组内训练延迟（$T_i = |D_i| / C_i$）尽可能接近。该策略通过排序后连续划分实现，有效降低组内同步延迟。

- **有界异步协调机制（Bounded Asynchronous Coordination）**  
  允许“活跃组”在满足最大延迟阈值 $T_{\text{max}}$ 的前提下，无需等待所有组即可参与全局更新。云服务器对过时的组信息进行控制，避免模型偏差过大。

- **三层架构支持（Cloud-Edge-Client）**  
  利用边缘层进行组内聚合，云端执行全局模型更新与对偶变量更新，形成高效协同优化流程。

### 🔍 相比现有方法的优势
| 方法 | 局限性 | P-GADMM 的改进 |
|------|--------|----------------|
| **标准 ADMM** | 同步阻塞，受最慢客户端拖累 | 引入异步机制，打破同步瓶颈 |
| **GADMM** | 分组依据非计算相关，组内仍有 straggler | 显式按计算能力分组，减少组内延迟差异 |
| **Asynch-ADMM** | 缺乏对 stale updates 的控制，收敛不稳定 | 设置 $T_{\text{max}}$ 控制延迟，保证收敛可靠性 |

> ✅ **综合优势**：P-GADMM 在保持 ADMM 良好收敛性的基础上，显著提升了训练效率，尤其适用于资源受限且高度异构的边缘环境。

---

## 2. 核心实验方法和设置

### 📊 数据集
- **MNIST**：手写数字识别任务，使用轻量 CNN 模型。
- **CIFAR10**：图像分类任务，采用更深的卷积神经网络（含 Dropout）。

### ⚙️ 实验设置
- **模拟平台**：基于 PyTorch 构建离散事件模拟器，真实反映计算延迟与等待时间。
- **客户端配置**：
  - 客户端数量：50
  - 组数：5（每组10个客户端）
  - 本地训练：1 epoch，batch size=64，学习率=0.01
- **异质性建模**：
  - **统计异质性**：通过 Dirichlet 分布控制标签分布偏移（$\alpha=100$: IID；$\alpha=0.1$: non-IID）
  - **系统异质性**：使用 Pareto 分布生成客户端响应延迟（shape=2.0: 轻尾；1.1: 重尾，straggler 更严重）

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Wall-clock time to target accuracy** | 达到目标准确率所需的真实时间（秒），为核心指标 |
| **Number of rounds** | 所需通信轮次 |
| **Average waiting time per round** | 每轮平均等待时间 |
| **Straggler blocking ratio** | 因等待慢组而被阻塞的比例 |
| **Final test accuracy** | 最终测试准确率 |
| **Convergence stability** | 多次运行下的标准差 |

### 🔁 基线方法对比
- **Asynch-ADMM** [32]：允许部分客户端更新，支持有界延迟的异步 ADMM。
- **GADMM** [26]：经典分组 ADMM，但分组随机或基于拓扑，无计算感知设计。

> 所有方法共享相同的数据划分、模型结构、超参数和随机种子，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Tables II & III）

#### ✅ MNIST 结果（target: 95% / 80%）
| Method       | Setting   | Rounds     | Time to Target (s) | Final Acc (%) |
|--------------|-----------|------------|---------------------|---------------|
| Asynch-ADMM  | IID       | 206.7±66.0 | 26.24±17.01         | 96.48±0.39    |
| GADMM        | IID       | 68.3±6.2   | 117.48±82.57        | 97.39±0.51    |
| **P-GADMM**  | IID       | 113.3±9.4  | **3.99±0.12**       | **98.00±0.26**|
| Asynch-ADMM  | non-IID   | 90.0*      | 25.27±3.75          | 83.70±3.79    |
| GADMM        | non-IID   | >500       | 2244.92             | 51.15±6.80    |
| **P-GADMM**  | non-IID   | 53.3±8.5   | **2.57±0.52**       | **94.77±0.27**|

> *仅 2/3 次运行达到目标

#### ✅ CIFAR10 结果（target: 75% / 60%）
| Method       | Setting   | Rounds      | Time to Target (s) | Final Acc (%) |
|--------------|-----------|-------------|---------------------|---------------|
| Asynch-ADMM  | IID       | 990.0±289.9 | 49.34±18.45         | 79.44±1.75    |
| GADMM        | IID       | >2000       | 1334.85             | 52.19±1.64    |
| **P-GADMM**  | IID       | 886.7±198.7 | **31.43±8.84**      | **81.27±1.62**|
| Asynch-ADMM  | non-IID   | 963.3±324.6 | 243.77±81.02        | 67.91±3.69    |
| GADMM        | non-IID   | >2000       | 6756.29             | 23.14±3.89    |
| **P-GADMM**  | non-IID   | 1070.0±215.2| **50.88±11.14**     | 66.69±3.66    |

### 🔍 对比分析
- **训练时间大幅缩短**：P-GADMM 在 MNIST 上比 GADMM 快 **30–900倍**，比 Asynch-ADMM 快 **6–10倍**。
- **最终精度更高或相当**：尤其在 non-IID 场景下，P-GADMM 显著优于 GADMM，且略优于或接近 Asynch-ADMM。
- **等待时间极低**：P-GADMM 平均每轮等待时间仅为 0.035–0.047 秒，远低于其他方法（最高达 4.5 秒）。
- **抗阻塞性强**：blocking ratio 显著低于 GADMM（后者常为 100%），说明其有效缓解了 straggler 影响。

### 🔧 消融实验（Sensitivity Analysis）
- **客户数量增加 → 收敛更快**：更多客户端带来更多并行组更新机会。
- **组尺寸增大 → 性能下降**：大组更易包含慢节点，降低更新频率。
- **延迟阈值 $T_{\text{max}}$ 增大 → 准确率下降**：
  - 当 $T_{\text{max}}=1$ 时，MNIST 最终准确率达 **94.75%**
  - 当 $T_{\text{max}}=10$ 时，降至 **92.08%**
  - 表明小延迟阈值有助于维持模型一致性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **计算感知分组显著提升效率**：将具有相似训练延迟的客户端聚类可有效减少组内同步开销。
2. **有界异步机制平衡效率与稳定性**：允许活跃组提前提交更新，同时通过 $T_{\text{max}}$ 控制 stale 信息影响，实现了高效且可靠的收敛。
3. **P-GADMM 在真实时间维度表现卓越**：尽管通信轮次不一定最少，但由于极低的等待时间和高并行度，**wall-clock training time 显著优于基线方法**。
4. **对 non-IID 和系统异质性鲁棒性强**：在极端异构环境下仍能稳定收敛，并取得更高最终精度。

### ⚠️ 方法的局限性
- **静态分组假设**：当前分组是静态的，未考虑客户端动态加入/退出或计算能力变化。
- **理想化收敛分析依赖强凸性**：理论分析基于 strongly convex objectives，实际深度学习任务多为非凸。
- **SGD 近似引入额外噪声**：实际实现中使用 local SGD 替代精确求解，会引入梯度方差，影响收敛边界。

### 🔮 未来工作方向
- 扩展至 **stochastic implementation** 的收敛分析。
- 设计 **adaptive grouping** 机制，动态调整组成员以应对网络波动。
- 探索 **dynamic $T_{\text{max}}$ 调整策略**，根据系统负载自动优化延迟容忍度。
- 应用于更复杂的联邦学习场景，如跨设备 FL 或垂直 FL。

---

> 💡 **总结一句话**：  
> **P-GADMM 通过“计算感知分组 + 有界异步协调”，在异构边缘网络中实现了 ADMM 的高效并行化，在显著降低 wall-clock training time 的同时，保持甚至提升了模型精度，为边缘智能提供了实用的分布式优化方案。**

</details>

---

### 6. [D-Quant: Driftable Entropy Coding for KV Cache Quantization](https://arxiv.org/abs/2609.19880)

**Authors**: Yi Su, Hong Liu, Guanghua Yu, Jianchen Zhu  
**Category**: cs.CL  
**Published**: 2026-09-18  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.19880v1  

#### Abstract
The KV cache has become a major bottleneck in deploying LLMs, as its memory footprint grows linearly with sequence length and batch size, imposing substantial pressure on both memory capacity and bandwidth. Among various KV cache compression techniques, quantization is particularly attractive due to...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# D-Quant: Driftable Entropy Coding for KV Cache Quantization 论文总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

在大语言模型（LLM）的自回归推理过程中，**KV Cache**（Key-Value Cache）是生成过程中的关键组件，用于存储已处理 token 的注意力 Key 和 Value 向量。然而，KV Cache 的内存占用随序列长度和 batch size 线性增长，成为部署长上下文应用时的主要瓶颈。

现有的 KV Cache 压缩方法多采用**固定位宽量化**（fixed-width quantization），例如 INT2 或 INT4，其存在两个根本缺陷：

- **表示能力受限**：b-bit 量化最多只能提供 $2^b$ 个量化等级，低比特下迅速丧失精度。
- **未利用分布非均匀性**：经过旋转（rotation）和归一化（normalization）后，KV 值近似服从正态分布，中心密集、尾部稀疏，但固定编码对高频和低频符号分配相同比特数，造成冗余。

此外，虽然熵编码（entropy coding）能有效压缩非均匀分布数据，但其**变长输出**难以适配高度并行的 attention kernel，导致无法直接集成。

---

### **提出了什么新方法或新思路**

本文提出 **D-Quant** —— 一种灵活的 KV Cache 量化框架，核心思想是：

> **将熵编码与固定大小存储容器结合，通过“漂移机制”（drift mechanism）实现可变长编码到定长比特流的转换。**

#### 主要技术亮点：

- ✅ **熵编码 + 固定容器设计**  
  对每个 token 的 Key/Value 流进行熵编码（使用 rANS），但强制所有流写入**固定字节长度的容器**（如 331 字节 Key + 231 字节 Value），从而保持内存布局规整，支持 fixed-stride 访问。

- ✅ **Drift Mechanism（漂移机制）**  
  当熵编码后的比特流超出预算时，并不降低整体量化等级，而是让少量高成本符号“漂移”至邻近低成本符号，以最小代价满足长度约束。该过程建模为带码长约束的率失真优化问题，用拉格朗日乘子法 + 二分搜索高效求解。

- ✅ **完全无需校准（calibration-free）的概率模型**  
  利用预处理（Hadamard rotation + mean removal）使 KV 分布逼近标准正态分布 $\mathcal{N}(0,1)$，直接从理论分布推导熵编码所需的概率表，无需额外校准数据。

- ✅ **灵活配置量化等级与比特预算**  
  不再受制于整数 bit-width 或 group size，可在相同平均比特下使用更多量化等级（如 8-level keys + 6-level values @ 2.26 BPV），实现更优率失真权衡。

---

### **相比现有方法的优势**

| 维度 | D-Quant | 传统量化方法 |
|------|--------|-------------|
| 编码效率 | 高（利用非均匀分布） | 低（等长编码） |
| 表示灵活性 | 强（M 可 ≠ $2^b$） | 弱（M = $2^b$） |
| 内存布局 | 规整（fixed-size 容器） | 规整 |
| 并行友好性 | 支持（单次解码 + warp 级操作） | 支持 |
| 是否需要校准 | ❌ 否（analytic model） | ✅ 是（empirical stats） |
| 性能损失 | 极小（接近 BF16） | 显著（尤其长上下文） |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **RULER**：评估长上下文理解能力，在 4K ~ 128K 不同 context length 下测试。
- **LongBench-E**：英文子集，涵盖多种任务（如 QA、摘要、代码等），用于综合性能评估。

---

### **实验设置和评估指标**

#### 模型
- **Qwen3-8B**
- **Llama-3.1-8B-Instruct**

#### 量化配置（D-Quant 默认）
- Key 容器：331 字节 → ~2.68 BPV
- Value 容器：231 字节 → ~1.84 BPV
- 平均：**2.26 bits per value (BPV)**
- 量化等级：Keys 使用 8-level，Values 使用 6-level
- Group size：1024（整个 token 作为一个 group）
- 最近 128 tokens 保留为 BF16（residual window）
- 概率锐化因子（sharpening factor）：$\alpha = 1.4$

#### 基线方法
| 方法 | 技术特点 | 存储开销 (BPV) |
|------|---------|----------------|
| **KIVI** | per-channel K, per-token V, INT2 | 2.25 |
| **QuaRot** | Hadamard rotation + INT2 | 2.25 |
| **TurboQuant** | Lloyd-Max + affine quantization | 2.19 |
| **OScaR** | Occam’s Razor design, INT2 | 2.26 |

所有 baseline 均使用 group size=128，且保留最近 128~256 tokens 为 BF16。

#### 评估指标
- **Accuracy**：RULER 和 LongBench-E 上的平均得分（越高越好）
- **Memory Footprint**：KV Cache 占用显存（越低越好）
- **Throughput**：解码吞吐量（tokens/sec，越高越好）
- **NMSE / KL Divergence**：重建误差与分布偏移（越低越好）

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

| 方法 | 存储 (BPV) | RULER (Qwen3-8B) | RULER (Llama-3.1-8B) | LongBench-E AVG |
|------|------------|------------------|------------------------|------------------|
| BF16 (Full Precision) | 16.00 | 87.31 | 89.69 | ~42.5 |
| **D-Quant** | **2.26** | **86.50** | **88.94** | **41.00 / 42.10** |
| D-Quant (更低预算) | 1.96 | 85.21 | 87.26 | — |
| OScaR | 2.26 | 78.09 | 85.01 | 40.67 / 41.87 |
| QuaRot | 2.25 | 79.21 | 65.39 | ~40.45 / 36.16 |
| KIVI | 2.25 | 65.88 | 74.43 | ~38.66 / 42.29 |

> 🔍 **观察**：D-Quant 在仅 **2.26 BPV** 下几乎无损地恢复 BF16 性能，显著优于所有 INT2 基线。

---

### **与基线方法的对比结果**

- 📈 **长上下文鲁棒性强**：随着 context length 增加（如 128K），传统方法性能急剧下降，而 D-Quant 仍紧贴 BF16 曲线。
  - 在 128K 上，D-Quant 得分 **70.92 vs. 最强 baseline 57.14**（Qwen3-8B）
- 💾 **KV Cache 内存减少高达 7×**
  - 在 64K 解码场景下，KV Cache 内存从 BF16 的数百 GiB 降至约 50 GiB。
  - Batch size=64 时，BF16 OOM，而 D-Quant 可扩展至 batch=128。
- ⚡ **解码吞吐提升达 3.5×**
  - 在最大可行 batch 下，D-Quant 达到 **563.6 tokens/s**，远超 BF16 的 159.5 tokens/s。
  - 虽然熵解码引入额外计算，但内存节省带来的并行收益更大。

---

### **消融实验结果**

#### （1）量化等级的影响（@ 固定 2.26 BPV）
- 使用 **8-level keys + 6-level values** 是最优配置。
- 进一步增加等级会导致 drift 过多，反而损害性能。

#### （2）存储预算的影响
- 增大容器尺寸持续提升准确率，逐渐趋近 BF16。
- 提供灵活的“存储-质量”权衡空间。

#### （3）KV 分配策略
- 将更多预算分配给 **Keys** 更有利（因其误差经 softmax 放大）。
- 但过度倾斜会牺牲 Values 表达能力，需平衡。
- 默认配置（keys 略多）为最佳折中。

#### （4）编码机制有效性验证（Appendix B）
- 控制其他条件一致，仅比较熵编码 vs 固定宽度编码：
  - D-Quant 的 KL 散度仅为 fixed-width 的 **~50%**
  - NMSE 下降 1.7~2.8×
  - 若要达到相同性能，fixed-width 需额外消耗 **+0.23~0.26 BPV**

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **固定位宽量化存在本质局限**：耦合了比特宽度与量化等级，无法充分利用 KV 值的高度非均匀分布特性。
2. ✅ **熵编码可用于 KV Cache 压缩**：只要通过“固定容器 + drift”机制解决变长输出问题，即可兼顾高压缩率与系统效率。
3. ✅ **分布建模可免校准**：经 Hadamard rotation + mean removal 后，KV 分布极接近标准正态分布，可直接构建解析概率模型。
4. ✅ **drift 是轻量且局部的操作**：平均仅 **0.75% keys 和 5.00% values** 被调整，且只移动一个 level，不影响整体稳定性。
5. ✅ **D-Quant 实现近乎无损压缩**：在 **2.26 BPV** 下保持接近 BF16 的 accuracy，同时带来 **7× 内存缩减** 和 **3.5× 吞吐提升**。

---

### **方法的局限性**

- 🔒 **依赖特定预处理流程**：必须先进行 Hadamard rotation 和 mean removal 才能使分布足够集中，否则熵模型失效。
- ⏱️ **编码延迟较高**：当前主要用于离线缓存压缩，实时 streaming 场景可能面临编码开销挑战。
- 🧩 **硬件适配尚未极致优化**：虽已设计专用 kernel，但在不同 GPU 架构上的性能仍有调优空间。

---

### **未来工作方向**

- 🔜 探索 **端到端训练感知的联合优化**：将 drift 和 entropy coding 纳入训练过程，进一步降低失真。
- 🔗 结合 **KV Cache Streaming**：与 CacheGen、SplitZip 等通信压缩方案协同，打造全链路高效 LLM serving 架构。
- 🖥️ 开发 **专用硬件加速器**：针对 rANS 解码 + drift 查表操作设计 ASIC/FPGA 模块，提升能效比。
- 🔄 动态容器大小调节：根据输入复杂度动态调整每 token 的容器大小，在极端长文本中实现更细粒度控制。

---

> ✅ **总结一句话**：  
> **D-Quant 成功弥合了熵编码的高压缩潜力与系统级高效推理之间的鸿沟，为 KV Cache 压缩开辟了一条兼具理论优雅性与工程实用性的新路径。**

</details>

---

### 7. [Agentic AI Networking for Heterogeneous Unmanned Aerial Systems in Low-Altitude Wireless Networks](https://arxiv.org/abs/2609.19538)

**Authors**: Nguyen Duc Minh Quang, Chang Liu, Shuangyang Li, Derrick Wing Kwan Ng  
**Category**: cs.AI  
**Published**: 2026-09-18  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.19538v1  

#### Abstract
Low-altitude wireless networks (LAWNs) are emerging as a key infrastructure for heterogeneous unmanned aerial systems that support concurrent services within a shared three-dimensional airspace. Their coexistence creates strong coupling among mobility, connectivity, and shared network resources, whi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Agentic AI Networking for Heterogeneous Unmanned Aerial Systems in Low-Altitude Wireless Networks*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在低空无线网络（**LAWN**）中，异构无人机系统（如物流、监控、中继等服务）共享有限的通信资源（带宽、地面基站容量、空域），导致以下挑战：
- 各服务独立优化自身目标，引发资源竞争与干扰；
- 服务需求和优先级动态变化，传统固定目标控制器难以自适应调整；
- 网络拓扑高度动态，要求低延迟、去中心化的控制机制。

现有方法（如静态优化、单任务学习控制器）无法同时满足**实时执行**与**高层策略自适应**的需求。

---

### 🚀 提出的新方法与创新思路
作者提出了一种**分层混合 LLM-MARL 架构**，采用**双闭环结构**（dual-loop structure）实现智能体化（agentic）协调：

#### （1）外环：**LLM-Assisted Game Orchestration**（战略编排）
- 利用 **Large Language Model (LLM)** 解释自然语言形式的服务需求、操作员意图、法规约束；
- 动态重构博弈参数（如效用权重 `w_m`、拥塞价格 `λ_r`），实现跨服务的目标重配置；
- 不依赖重新训练即可适应新场景。

#### （2）内环：**Parameter-Conditioned MARL Execution**（分布式执行）
- 多智能体强化学习（**MARL**）代理基于本地观测 + 共享信道知识图谱（**CKM**）做出实时通信与移动决策；
- 政策为“参数条件化”（parameter-conditioned），可响应外环广播的游戏参数而无需再训练。

#### （3）共享记忆：**Channel Knowledge Map (CKM)**
- 地面基站聚合各无人机的稀疏测量，生成密集的无线电环境地图；
- 所有智能体可查询 CKM 获取未见区域的信道状态，缓解部分可观测性问题。

> 🔍 **核心创新**：首次将 LLM 的语义理解能力与 MARL 的实时决策能力结合，形成一个**自主闭环的 agentic 系统**，支持异构服务在动态环境中协同共存。

---

### ⚖️ 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| 自适应性 | 固定目标，需离线重训练 | 运行时通过 LLM 动态重配置目标 |
| 协调方式 | 集中式或无协调 | 分布式执行 + 战略级编排 |
| 决策灵活性 | 缺乏高层推理 | LLM 可处理自然语言指令与突发事件 |
| 可扩展性 | 中心化开销大 | 去中心化 MARL + 层次化管理提升可扩展性 |

---

## 2. 核心实验方法和设置

### 🧪 实验场景设计
- **城市区域**：约 1 km²，部署 3 个 **GBS**（Ground Base Stations）；
- **两类 UAV 智能体**：
  - **Logistics Swarm**（5架）：执行仓库到客户的包裹投递，依赖可靠控制信号；
  - **Monitoring Swarm**（5架）：缓慢巡航于指定区域，维持持续覆盖；
- **共享资源**：GBS 上的资源块（Resource Blocks, RBs），按 5G NR 等分分配。

---

### 📊 评估指标
- **Joint Utility**：归一化后的总效用 $\sum_m u_m$，以集中式最优解为基准（值为 1.0）；
- **Individual KPIs**：
  - 物流：交付效率（delivery efficiency）；
  - 监控：覆盖率持久性（coverage persistence）；
- **Normalized Coverage / Delivery Rate**：相对于集中式参考的性能比例；
- **Training Curves**：联合效用随训练步数的变化趋势。

---

### 🔁 基线方法对比
| 基线名称 | 描述 |
|--------|------|
| **Selfish Operation** | 各 swarm 独立最大化自身效用，无跨服务协调 |
| **Centralized Benchmark** | 全局信息下的集中式优化结果（理论上限） |
| **MARL without Reconfiguration** | MARL 政策固定训练时的参数，无法运行时调整 |
| **LLM-Orchestrated Greedy** | LLM 编排目标，但执行层使用贪心策略而非 MARL |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Fig. 4 和 Fig. 5）

#### （1）协调效果显著优于自私行为（Fig. 4a）
- **Selfish Baseline**：两服务均收敛至同一 GBS，造成严重拥塞；
  - 联合效用仅达集中式基准的 ~55%；
- **Proposed Method**（LLM+MARL）：
  - 达到集中式基准的 **~85%**；
  - 引入拥塞定价后，智能体主动避开繁忙 GBS，实现资源均衡利用；
  - → **联合性能提升超过 50%**（相对自私策略）；

#### （2）共享 CKM 显著提升性能（Fig. 4b，消融实验）
- 移除共享 CKM，仅使用本地感知地图：
  - 物流 KPI 下降 **20%**；
  - 监控 KPI 下降 **30%**；
- 原因：缺乏全局视野导致路径规划次优，尤其对需要长期覆盖的任务影响更大。

#### （3）训练稳定性高（Fig. 4c）
- 提出的方法训练曲线平滑收敛；
- Selfish baseline 波动剧烈，反映对抗性竞争带来的不稳定性。

#### （4）运行时自适应能力强（Fig. 5）
模拟事件：第 70 轮发生公共安全事件，需优先保障监控服务。
- **Configuration A → B**：
  - LLM 将 `w_monitoring` 从 1 提升至 2，`λ_1` 提升至 1.5；
- 结果：
  - **LLM-orchestrated MARL**：快速恢复至接近新最优水平（~0.85）；
  - **Fixed MARL**：始终停留在 ~0.55，无法适应新目标；
  - **Greedy Policy**：虽接受新目标，但因忽略未来影响和对手反应，性能震荡且最终低约 0.2；
- ✅ 表明：**只有 LLM 编排 + MARL 执行的组合才能实现稳定高效的动态适应**。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **异构服务共存应建模为动态非合作博弈**，其目标函数需随时间演进；
2. **LLM 适合作为高层“战略编排器”**，能够解析自然语言指令并生成合理的博弈参数配置；
3. **Parameter-conditioned MARL 支持零样本运行时适应**，无需重新训练即可响应新目标；
4. **共享 CKM 是缓解部分可观测性的关键基础设施**，促进跨服务知识共享；
5. **LLM + MARL 的混合架构实现了“感知-记忆-推理-行动”（PMRA）闭环**，是构建 agentic 网络的有效范式。

---

### ⚠️ 方法的局限性
| 限制 | 说明 |
|------|------|
| **LLM 推理延迟与成本** | LLM 推理耗时较长（百毫秒至秒级），不适合高频控制；目前仅用于慢速编排环 |
| **仿真到现实差距（Sim-to-Real Gap）** | 当前实验基于理想化模型，真实飞行中的动力学、天气等因素尚未充分建模 |
| **CKM 安全与完整性风险** | 共享地图易受恶意注入或过时数据攻击，需引入可信机制 |
| **可扩展性挑战** | 随着 swarm 规模扩大，验证博弈稳定性难度增加，LLM 配置复杂度上升 |

---

### 🔮 未来研究方向（原文第五节）
1. **Scale-out 架构设计**：
   - 引入图神经网络或分段专用 LLM 降低编排负担；
2. **高保真数字孪生（Digital Twin）**：
   - 构建融合空气动力学、电池、天气的真实仿真环境；
3. **安全可信的 CKM 构建机制**：
   - 结合加密认证与不确定性建模，确保地图完整性；
4. **天地一体化网络集成**：
   - 利用卫星提供广域校准，支持多尺度联邦学习融合；
5. **标准化跨平台接口**：
   - 推动 GUTMA 等组织制定统一的数据交换格式与协议，实现互操作性。

---

## 总结

> 本论文开创性地提出了 **agentic AI networking** 在异构低空网络中的应用框架，通过 **LLM-MARL 双环架构** 实现了“高层语义理解”与“底层实时控制”的有机统一。实验证明该方法不仅能有效缓解资源竞争，还能在突发事件下自主重配置目标，展现出强大的适应性与鲁棒性，为未来自治型低空经济网络提供了重要技术路径。

</details>

---

### 8. [MetaRTL: Meta-path Attention Enhanced Relational Table Learning](https://arxiv.org/abs/2609.19832)

**Authors**: Ken Zhong, Weichen Li, Zheng Wang  
**Category**: cs.AI  
**Published**: 2026-09-18  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.19832v1  

#### Abstract
Relational table learning has gained increasing attention with the widespread use of relational databases. Existing methods typically rely on deep GNN or HGNN stacks, leading to high computational costs and limited performance on large real-world databases. We propose MetaRTL, a two-stage framework ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# MetaRTL: Meta-path Attention Enhanced Relational Table Learning 论文总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

现有的 **Relational Table Learning (RTL)** 方法通常依赖于深层的 **GNN** 或 **HGNN** 架构来建模多表关系数据库中的复杂结构。然而，这类方法面临以下挑战：

- **高计算成本**：在大规模真实世界数据库上训练深层 GNN/HGNN 非常耗时且资源密集。
- **过平滑（over-smoothing）** 和 **过拟合** 问题：随着网络加深，节点表示趋于相似，导致表达能力下降。
- **对采样敏感**：基于邻居采样的训练方式可能引入语义偏差。

此外，许多现实数据库具有高度复杂的连接结构（如 Stack Exchange 数据集），使得传统方法难以高效扩展。

---

### **提出了什么新方法或新思路**

作者提出 **MetaRTL** —— 一种两阶段、可扩展且表达能力强的 RTL 框架，其核心思想是：

> 将计算重心从“深度参数化消息传递”转移到“轻量级元路径特征聚合”。

#### **两个阶段设计如下：**

1. **Pre-training Stage（预训练阶段）**
   - 使用一组 **TNNs** 编码各表行数据，生成初始节点嵌入。
   - 联合一个浅层（仅两层）的 **HGNN** 进行极短周期（实验中为 5 轮）训练。
   - 目标不是完全收敛，而是获得稳定、语义丰富的初始表示。

2. **Aggregation Stage（聚合阶段）**
   - 利用非参数化传播（non-parametric propagation）沿预定义的 **meta-path** 提取高阶语义特征。
   - 引入 **MetaAttn** 模块进行融合：
     - **Meta-path Self-Attention (MPSA)**：动态加权不同 meta-path 特征的重要性。
     - **Global Node Cross-Attention (GNCA)**：通过全局聚类中心缓解采样带来的信息丢失。
   - 最终输出任务特定的目标节点表示。

---

### **相比现有方法的优势**

| 维度 | MetaRTL 的优势 |
|------|----------------|
| **效率** | 避免深层 GNN 堆叠，训练时间随节点数近似线性增长，显著提升可扩展性。 |
| **表达能力** | 利用 meta-path 显式捕捉跨表复杂语义路径，增强模型解释性和语义感知能力。 |
| **鲁棒性** | 通过 GNCA 引入全局上下文，减轻邻居采样带来的偏差；避免过平滑。 |
| **灵活性** | 支持 mini-batch 训练，适用于超大图场景。 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

在两个公开的 RTL benchmark 上进行全面评估：

#### **SJTUTables**
- 包含 3 个中小规模多分类任务：
  - `TACM12k`, `TLF2K`, `TML1M`
- 表数量少（3–4）、关系较稀疏。

#### **RelBench**
- 包含 7 个大规模真实世界数据集，涵盖电商、社交网络、医疗等领域。
- 共 **21 个任务**（12 个二分类 + 9 个回归）
- 数据规模巨大（最大达上亿条记录），更贴近实际应用。

| Dataset | #Tables | Max Rows | Task Type |
|--------|---------|----------|-----------|
| rel-f1 | 9 | ~97k | Classification / Regression |
| rel-trial | 15 | ~5.8M | ... |
| rel-avito | 8 | ~20.6M | ... |
| rel-amazon | 3 | ~24.2M | ... |
| rel-hm | 3 | ~33.2M | ... |
| rel-stack | 7 | ~38.1M | ... |
| rel-event | 5 | ~41.3M | ... |

---

### **实验设置和评估指标**

- **采样策略**：采用 **temporal uniform neighbor sampling** 进行 mini-batch 训练。
- **评估协议**：严格遵循官方 benchmark 设置（来自 RelBench 和 SJTUTables）。
- **评价指标**：
  - 分类任务：**ROC-AUC**
  - 回归任务：**MAE**
  - 综合比较：**平均排名（average ranking）**

---

### **基线方法对比**

分为两类：

#### **Single-table Methods**
- **LightGBM**：梯度提升树基线
- **FTTransformer**：基于 Token 的 Transformer 模型
- **Trompt**：提示式 tabular 模型
- **ExcelFormer**：增强列间交互的 attention 模型

> 注：这些方法需先将辅助表通过 left join 合并到目标表。

#### **Multi-table Methods**
- **RDL**：结合 ResNet-TNN 与 GraphSAGE 的通用框架
- **BRIDGE**：简化版 GCN，仅支持双表关系
- **RelGNN**：基于原子路径分解的消息传递
- **LightRDL**：蒸馏小 GNN + 时间图结构，强调推理效率

> 作者复现了部分未开源的方法，并对 BRIDGE 和 LightRDL 做了合理适配。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ 在 **SJTUTables** 上的结果（Accuracy %）

| Method | TACM12k | TLF2K | TML1M |
|-------|--------|-------|-------|
| LightGBM | 35.70 | 35.52 | 26.66 |
| BRIDGE | 25.60 | 42.20 | 36.20 |
| **MetaRTL** | **46.54** | **43.17** | **37.09** |

- MetaRTL 在所有三个任务上均取得最优结果。
- 在 TACM12k 上比 LightGBM 提升 **+11%**，比最强深度学习基线 BRIDGE 提升 **+21%**。

> 原因分析：MetaRTL 更好地利用了紧凑但信息丰富的表结构，避免了 RDL/RelGNN 在简单结构上的过拟合。

---

#### ✅ 在 **RelBench** 上的表现（分类任务：ROC-AUC；回归任务：MAE）

##### **分类任务汇总（Table III）**
- MetaRTL 在 **12 项分类任务中的 9 项**达到最佳。
- 平均排名：**1.33**（远优于第二名 RDL 的 2.92）

##### **回归任务汇总（Table IV）**
- MetaRTL 在 **9 项回归任务中的 9 项**表现最好或接近最优。
- 平均排名：**1.67**（优于所有基线）

> 示例亮点：
> - `rel-amazon item-ltv`：MAE = **46.176**（vs. 第二名 48.112）
> - `rel-hm item-sales`：MAE = **0.039**（vs. 第二名 0.044）
> - `rel-event user-attendance`：MAE = **0.241**（持续领先）

✅ 总体结论：**MetaRTL 在 21 个任务中拿下 18 项第一，平均排名稳居首位。**

---

### **消融实验结果（Ablation Study）**

在 `rel-f1` 数据集上验证各组件作用：

| Variant | driver-top3↑ | driver-dnf↑ | driver-position↓ |
|--------|--------------|------------|------------------|
| w/o TNN Pretrain | 0.7311 | 0.6761 | 4.319 |
| w/o Meta-path Features | 0.6294 | 0.6013 | 4.870 |
| w/o Global Attention | 0.8389 | 0.7216 | 4.110 |
| **MetaRTL (Full)** | **0.8415** | **0.7418** | **3.987** |

#### 发现：
- 移除 **TNN 预训练** 导致性能大幅下降 → 初始嵌入质量至关重要。
- 移除 **meta-path 特征** 影响最大（尤其 regression）→ 高阶语义路径不可替代。
- 移除 **global attention** 也有轻微影响 → 全局上下文有助于缓解采样偏差。

---

## 4. 关键结论和发现

### **主要发现**

1. **Meta-path 是建模复杂关系的有效手段**  
   显式利用 schema-level meta-path 可以有效捕获跨多个辅助表的语义依赖，优于隐式的多跳消息传递。

2. **两阶段解耦设计优于端到端深层堆叠**  
   浅层预训练 + 非参数化 meta-path 聚合，在保持高效的同时提升了表达能力和泛化性。

3. **MetaAttn 模块实现了高效而灵活的特征融合**  
   - MPSA 实现 meta-path 级别注意力选择。
   - GNCA 引入全局语义补偿局部采样损失。

4. **MetaRTL 具备良好的可扩展性**  
   - 图 5 显示训练时间随节点数呈近似线性增长。
   - 非参数 meta-path 计算开销极低，适合大规模部署。

5. **预训练不宜过长**  
   - 图 4 显示：预训练超过 10 轮后性能略有下降 → 存在过拟合风险。
   - “轻量预训练 + 强聚合” 是更优范式。

---

### **方法的局限性**

- **依赖人工定义 meta-path**：虽然 meta-path 来自 schema，但仍需领域知识或启发式规则设定。
- **对非常深的关系（>3 hop）建模有限**：当前 meta-path 长度受限于实际计算可行性。
- **不适用于无明确 pkey-fkey 结构的数据源**：如自由文本或图谱缺失场景。

---

### **未来工作方向**

- 扩展至 **data lake 场景**（文中已提及）：整合半结构化与非结构化数据。
- 自动挖掘重要 meta-path 或实现 **meta-path 学习机制**。
- 探索与 **LLM-based RTL** 方法（如 TLLM）的结合路径。
- 支持动态更新与流式推理，适应实时业务需求。

---

## ✅ 总结

**MetaRTL** 是一项针对大规模关系表学习任务提出的高效、可解释、高性能的新框架。它通过 **“轻预训练 + 强聚合”** 的两阶段设计，成功平衡了模型表达力与计算效率之间的矛盾，在 10 个真实世界数据集、24 项任务中全面超越现有 SOTA 方法，为未来工业级 RTL 系统提供了新的技术范式。

</details>

---

### 9. [Xronos: Heterogeneity-Aware Tensor Parallelism for Collaborative LLM Fine-Tuning on Edge CPUs](https://arxiv.org/abs/2609.19909)

**Authors**: Wonmi Choi, Sunjae Park, Dohyeok Kwon, Zhixiong Niu, Yeonho Yoo, Chuck Yoo, Gyeongsik Yang  
**Category**: cs.DC  
**Published**: 2026-09-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.19909v1  

#### Abstract
Collaborative fine-tuning on edge devices adapts large language models to domain-specific data while keeping each device's data local. State-of-the-art (SOTA) collaborative fine-tuning techniques are largely designed for GPU-based edge devices and rely on pipeline parallelism (PP). However, many edg...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# XRONOS: Heterogeneity-Aware Tensor Parallelism for Collaborative LLM Fine-Tuning on Edge CPUs 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文针对**边缘设备上的协作式大语言模型（LLM）微调**场景，指出当前主流方法在 **CPU-based 边缘设备** 上存在严重性能瓶颈：

- **Pipeline Parallelism (PP)** 是现有 SOTA 协同训练框架（如 Asteroid）广泛采用的技术，其依赖计算与通信的重叠来提升效率。
- 然而，在 **CPU-based 边缘设备**（如 IoT 网关、智能家居中枢、车载计算机）上，**同一 CPU 需同时处理模型计算和通信任务**，导致严重的资源争用（CPU contention），使得 PP 的“重叠”优势失效。
- 此外，现有 **Tensor Parallelism (TP)** 方法（如 Megatron-LM）假设设备同构（homogeneous），无法适应真实边缘环境中设备算力差异大的情况，造成“straggler”问题 —— 快速设备大量空闲等待慢速设备同步。

### 提出了什么新方法或新思路
作者提出 **XRONOS**，一个专为异构 CPU 边缘设备设计的协同微调框架，核心思想如下：

- **以 TP 作为执行主干**：避免 PP 中因共享 CPU 导致的计算-通信争用问题。
- **引入轻量级性能剖析（lightweight profiling）**：仅对少量代表性层（embedding、一个 transformer block、output layer）进行微调测试，快速估计各设备的计算延迟和内存占用。
- **异构感知的张量划分策略（heterogeneity-aware tensor partitioning）**：
  - 基于 profiling 结果，动态决定每个 worker 应分配多少 partition units（如注意力头数）。
  - 通过优化搜索算法，在满足各设备内存限制的前提下，最小化最慢 worker 的本地微调时间，从而减少整体迭代时间和设备空闲率。

### 相比现有方法的优势
- **避免 CPU contention**：相比 PP，显著降低 context switch 和 computation stall ratio。
- **缓解 straggler 问题**：相比传统均匀划分的 TP，大幅减少快速设备的 idle time。
- **高效实用**：profiling 开销小，适用于实际部署；支持异构设备组合，更具现实意义。
- **保持精度不变**：在加速的同时不牺牲模型最终准确率。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **GLUE benchmark** 中的三个下游任务：
  - **CoLA**（语言可接受性判断）
  - **SST-2**（情感分析）
  - **MRPC**（释义检测）

### 实验设置
#### 模型
- **RoBERTa-Base**（125M 参数）
- **GPT-2-Medium**（345M 参数）
- **MobileLLaMA-1.4B**（1.4B 参数）

| Model             | Arch.   | Params | #Transformer blocks |
|------------------|---------|--------|---------------------|
| RoBERTa-Base     | Encoder | 125M   | 12                  |
| GPT-2-Medium     | Decoder | 345M   | 24                  |
| MobileLLaMA-1.4B | Decoder | 1.4B   | 24                  |

#### 设备配置（异构组合）
所有设备通过 1Gbps 以太网连接。

| Device         | CPU Processor           | Memory |
|---------------|--------------------------|--------|
| Raspberry Pi 5 | ARM A76 (2.4GHz)        | 8 GB   |
| Orange Pi 5+   | ARM A76/A55 (2.4/1.8GHz)| 16 GB  |
| LattePanda Mu  | Intel N100 (3.4GHz)     | 8 GB   |
| ASUS MiniPC    | Intel N100 (3.4GHz)     | 16 GB  |

**实验场景**：
- **Set-A**: 1× ASUS MiniPC + 2× Raspberry Pi 5
- **Set-B**: 1× LattePanda Mu + 1× Orange Pi 5+ + 2× Raspberry Pi 5
- **Set-C**: 1× ASUS MiniPC + 1× LattePanda Mu + 1× Raspberry Pi 5 + 2× Orange Pi 5+

#### 评估指标
- **Iteration time**：单次迭代平均耗时（越低越好）
- **Computation stall ratio**：CPU stall cycles 占比（反映资源争用）
- **Number of context switches**：上下文切换次数（反映调度开销）
- **Idle time ratio**：设备在同步点等待的时间占比（衡量负载均衡）
- **Time-to-accuracy**：达到目标准确率所需时间
- **Estimation error**：微调时间预测误差（MAPE）

### 基线方法对比
- **Asteroid [5]**：基于 PP 的 SOTA 协同训练框架，支持异构设备规划。
- **Megatron-LM [15]**：经典 TP 框架，用于 GPU 集群，采用均匀张量划分，不考虑异构性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### 迭代时间（Iteration Time）
XRONOS 在所有任务和设备组合中均取得最优表现：

| Model            | vs. Asteroid (Avg.) | vs. Megatron-LM (Avg.) |
|------------------|----------------------|-------------------------|
| RoBERTa-Base     | ↓ **31%** (最高 ↓53%) | ↓ **14%** (最高 ↓18%)   |
| GPT-2-Medium     | ↓ **28%** (最高 ↓43%) | ↓ **14%** (最高 ↓18%)   |
| MobileLLaMA-1.4B | ↓ **39%** (最高 ↓56%) | ❌ OOM（内存溢出）       |

> 注：Megatron-LM 因未考虑内存异构性，在多个场景下出现 OOM。

#### 计算停滞比（Computation Stall Ratio）
XRONOS 显著低于 Asteroid（PP-based），表明有效缓解了 CPU 资源争用：

- RoBERTa-Base: ↓ **20%**
- GPT-2-Medium: ↓ **23%**
- MobileLLaMA-1.4B: ↓ **18%**

#### 上下文切换次数（Context Switches）
相比 Asteroid，XRONOS 减少幅度达：
- RoBERTa-Base: ↓ **98.82%**
- GPT-2-Medium: ↓ **96.73%**
- MobileLLaMA-1.4B: ↓ **96.86%**

#### 设备空闲率（Idle Time Ratio）
相比 Megatron-LM，XRONOS 极大降低了快速设备的等待时间：

| Model            | Megatron-LM (Avg.) | XRONOS (Max.) | Reduction |
|------------------|--------------------|---------------|-----------|
| RoBERTa-Base     | ~18%               | <6%           | **4.6×**  |
| GPT-2-Medium     | ~26%               | <6%           | **5.9×**  |

#### 时间-精度分析（Accuracy-Time Analysis）
在 RoBERTa-Base + CoLA + Set-A 场景下，以 90% 准确率为目标：
- XRONOS 达到目标时间比 Asteroid **快 53.6%**
- 比 Megatron-LM **快 16.4%**

#### 微调时间预测误差（Micro-benchmark）
- 平均 MAPE 误差为 **15.32%**（最低 13.28%，最高 16.65%），说明 profiling 预测具有较高可靠性。

---

## 4. 关键结论和发现

### 主要发现
1. **PP 不适合 CPU-based 边缘设备**：由于 CPU 同时承担计算与通信，PP 的“重叠”机制反而引发严重资源争用，导致平均 **5.75× 更高的 computation stall ratio**。
2. **TP 更适合作为 CPU 边缘设备的执行范式**：因其将计算与通信分离，天然更适合单处理器环境。
3. **现有 TP 方法在异构边缘环境下效率低下**：均匀划分导致快速设备最多 **34% 的时间处于 idle 状态**。
4. **XRONOS 有效解决了上述问题**：
   - 采用 TP 避免 CPU contention；
   - 引入异构感知划分策略，实现负载均衡；
   - 实现端到端 **最高 56% 的迭代加速** 和 **~5.9× 的 idle time 降低**。

### 方法的局限性
- **Profiling 假设稳定性**：假设设备性能在训练过程中稳定，未考虑温度 throttling 或系统负载波动的影响。
- **静态划分策略**：partition ratio 在训练开始前确定，未支持运行时动态调整。
- **仅支持同步 TP**：未探索异步或弹性训练机制。
- **网络带宽假设一致**：虽然文中提到通信时间由最慢带宽决定，但未显式建模高延迟链路的影响。

### 未来工作方向
- 支持 **动态 workload rebalancing**，应对运行时性能变化。
- 探索 **混合并行策略**（如 TP + DP）以进一步扩展规模。
- 将 XRONOS 扩展至 **GPU+CPU 混合边缘集群**。
- 结合 **energy efficiency** 优化，面向低功耗边缘场景。
- 探索 **联邦学习 + XRONOS** 的集成方案，增强隐私保护能力。

--- 

> ✅ 总结：XRONOS 是首个系统性解决 **异构 CPU 边缘设备上 LLM 协同微调效率问题** 的框架，通过 **TP + lightweight profiling + heterogeneity-aware planning** 的组合拳，在真实设备上实现了显著的性能提升，为边缘 AI 的落地提供了重要技术路径。

</details>

---

### 10. [Distributed Edge Inference: an Experimental Study on Multiview Detection](https://arxiv.org/abs/2609.20009)

**Authors**: Gianluca Mittone, Giulio Malenza, Marco Aldinucci, Robert Birke  
**Category**: cs.DC  
**Published**: 2026-09-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.20009v1  

#### Abstract
Computing is evolving rapidly to cater to the increasing demand for sophisticated services, and Cloud computing lays a solid foundation for flexible on-demand provisioning. However, as the size of applications grows, the centralised client-server approach used by Cloud computing increasingly limits ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Distributed Edge Inference: an Experimental Study on Multiview Detection*

---

## 1. **论文的主要贡献和创新点**

### 解决的问题
本文针对当前 **Cloud Computing** 在处理大规模、高资源消耗的 AI 应用（如基于 Transformer 的大模型）时面临的 **可扩展性瓶颈**，探索如何利用 **Compute Continuum**（计算连续体，涵盖 Cloud/Edge/Fog）实现更高效的分布式推理。特别关注在边缘设备上部署复杂的多视角检测（Multiview Detection, MvDet）系统所面临的挑战。

传统方法通常采用集中式 client-server 架构或将单个 DNN 部署到边缘设备进行优化（如压缩、蒸馏），但缺乏对 **跨多个边缘节点协同执行模型** 的系统性研究。

### 提出的新方法与新思路
- **实现了首个基于真实世界场景的分布式边缘推理系统**，用于 multiview detection，基于先进的 MVDet 模型。
- 将 MVDet 模型从原始 Python 实现移植为 **C++ 实现**，集成到高性能分布式框架 **FastFL** 中，支持低开销、高效率的边缘部署。
- 设计并实现了两种部署架构：
  - **Centralised Implementation**：所有计算集中在 Aggregator 节点，摄像头仅负责采集帧。
  - **Distributed Implementation**：将模型划分为两部分——**feature extraction + perspective warp** 分布到各 Camera 节点执行，**spatial aggregation** 由 Aggregator 完成，充分利用边缘算力。
- 利用 **FastFlow** 后端构建灵活的树状通信拓扑（tree-based topology），突破主流 DML 框架对 master-worker 结构的限制。

### 相比现有方法的优势
- **更高的推理效率**：在合适条件下，分布式方案相比集中式实现最高达 **1.92x 的推理速度提升**。
- **更好的资源利用率**：通过模型划分和并行化，有效利用边缘侧闲置算力，缓解中心节点压力。
- **灵活性更强**：支持自定义通信拓扑和轻量级运行时，更适合异构、资源受限的边缘环境。
- **开源实现**：提供一个可用于其他模型迁移的参考架构和方法论。

---

## 2. **核心实验方法和设置**

### 使用的数据集
- **Wildtrack Multiview Dataset**  
  包含 7 个固定摄像头拍摄的公共开放区域行人视频，提供时间同步的 Full-HD 帧（共 400 帧）、精确相机位置与视角参数，适合 multiview detection 场景验证。

### 实验设置
- **测试平台**：基于 **HPC4AI** 云计算设施构建虚拟化测试床，使用 10 台虚拟机，每台配置：
  - 8 个 64-bit vCPU（Intel Xeon Gold-6230 @2.10GHz）
  - 16GB RAM
  - 1 Gb/s 网络互联
  - Ubuntu 22.04
- 所有组件（7 个 Camera、1 个 Sync、1 个 Aggregator、1 个 ControlRoom）分别部署在独立 VM 上。
- 使用 **CPU-only** 设置以更好模拟典型边缘设备能力（无 GPU）。
- 使用 `taskset` 和环境变量（`MKL_NUM_THREADS`, `OMP_NUM_THREADS`）控制各节点可用核心数。

### 评估指标
- **主要性能指标**：处理一整套 7 个摄像头帧并输出最终位置估计所需的时间（seconds per frame set）。
- 每组实验重复 5 次，报告均值 ± 95% 置信区间（CI）。

### 基线方法对比
- **Centralised Baseline**：代表传统的 client-server 模式，所有计算负载集中在 Aggregator。
- **Proposed Distributed Approach**：作为对比方案，体现去中心化带来的潜在收益。

此外还通过调节以下因素进行消融分析：
- 各 Camera 和 Aggregator 的 **计算资源分配**（1/2/4/8 cores）
- 网络带宽条件（模拟不同网络环境）

---

## 3. **主要实验结果和性能指标**

### 关键性能数据
#### 不同计算资源配置下的表现（见 Figure 2）
| 配置 | Centralised 平均耗时 (s/set) | Distributed 平均耗时 (s/set) |
|------|-------------------------------|------------------------------|
| 最优集中式（8c Agg） | ~7.66 s | — |
| 最优分布式（8c Cam, 8c Agg） | — | **4.23 s** |
| 最差分布式（1c Cam, 4c Agg） | — | 12.97 s |

> 分布式方案随着边缘设备算力增强显著提升性能，最大提速达 **1.92x**。

#### 不同网络带宽下的性能对比（见 Table 1）
| Aggregator Cores | Camera BW (up/down) | Aggregator BW | Centralised (s/set) | Distributed (s/set) | Speedup (Centralised / Distributed) |
|------------------|---------------------|---------------|--------------------|----------------------|-------------------------------------|
| 4 cores | 25/284 Mb/s | 1/1 Gb/s | 15.38 | 49.68 | **3.23x** in favor of centralised |
| 8 cores | 25/284 Mb/s | 1/1 Gb/s | 11.98 | 46.89 | **3.91x** |
| 8 cores | 10/10 Mb/s | 100/100 Mb/s | 14.22 | 108.79 | **7.65x** |

> 在低带宽环境下，由于需传输更大的 **feature maps**（21.6MB vs 原始帧 6.1MB），分布式方案通信开销剧增，导致性能远劣于集中式。

### 与基线方法的对比结果
- 在 **高计算资源 + 高带宽** 条件下，**distributed 方案明显优于 centralised**，最高快 **1.92x**。
- 在 **低带宽或边缘设备算力不足** 时，centralised 方案反而更具优势，因其通信负载更小。
- 表明：**没有绝对最优策略**，系统设计必须权衡计算与通信成本。

### 消融实验结果
- **计算资源影响**：
  - Centralised 对 Camera 算力不敏感；性能主要取决于 Aggregator 算力。
  - Distributed 性能随 Camera 和 Aggregator 算力共同提升而持续改善。
- **网络带宽影响**：
  - 分布式对带宽极为敏感，是其性能瓶颈之一。
  - 当网络不再是瓶颈时，分布式优势全面显现。

---

## 4. **关键结论和发现**

### 主要发现
1. **分布式边缘推理具有巨大潜力**：在理想条件下（充足边缘算力 + 高速网络），通过合理划分模型任务，可以显著提升推理效率，实现高达 **1.92x 的加速**。
2. **环境因素决定架构选择**：实际部署中，**计算能力分布** 与 **网络带宽** 是决定 centralised 还是 distributed 更优的关键因素。
3. **通信开销不可忽视**：尽管边缘设备具备一定算力，但传输中间特征图（feature maps）的成本可能抵消本地计算节省的时间，尤其在 4G/低速 5G 等受限网络中。
4. **现有 DML 框架不适合边缘场景**：主流框架（Python-based, master-worker only）在资源效率和通信拓扑灵活性方面无法满足边缘需求，因此作者选用 **FastFL (C/C++)** 自主构建系统。

### 方法的局限性
- **未优化通信**：直接传输原始 feature maps，未采用压缩、量化等手段降低通信负载。
- **依赖预设 homography 矩阵**：假设相机标定已知且稳定，动态变化场景适应性有限。
- **仿真环境限制**：实验基于虚拟机模拟，未完全反映真实边缘设备的功耗、热节流等问题。
- **模型未轻量化**：使用完整 ResNet18，未结合 distillation 或 pruning 技术进一步适配边缘设备。

### 未来工作方向
- 引入 **model compression**, **quantization**, **feature map compression** 等技术降低通信开销。
- 探索 **adaptive runtime scheduler**，能够根据实时网络与计算状态动态切换 centralised/distributed 模式。
- 利用专用硬件加速器（如 **TPU**, **NPU**, **FPGA**）进一步提升边缘节点性能。
- 扩展至更多类型的 DNN 模型和应用场景，验证方法通用性。
- 在真实边缘设备（如 Raspberry Pi, Jetson Nano）上部署验证端到端性能与能耗。

--- 

> ✅ **总结一句话**：  
> 该论文通过实证研究表明，在 compute continuum 环境下，**分布式边缘推理能否胜出取决于“计算-通信”权衡**；未来需要构建 **动态适应环境变化的智能调度机制** 才能真正释放其潜力。

</details>

---

### 11. [dQwen3.5: Hybrid-Attention Diffusion Language Models](https://arxiv.org/abs/2609.20751)

**Authors**: Anton Xue, Litu Rout, Aditya Akella, Adam Klivans, Sujay Sanghavi, Sanjay Shakkottai  
**Category**: cs.CL  
**Published**: 2026-09-18  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.20751v1  

#### Abstract
Adapting a pretrained autoregressive (AR) model is a cost-efficient route to a diffusion language model (DLM). While nearly all such adaptations start from a full-attention transformer, AR modeling has shifted toward hybrid architectures that interleave attention and RNN layers. This creates an obst...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：dQwen3.5: Hybrid-Attention Diffusion Language Models

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

当前主流的 **Diffusion Language Models (DLMs)** 多基于全注意力（full-attention）架构的自回归（AR）模型进行适配（adaptation）。然而，现代高性能语言模型正转向**混合架构（hybrid architecture）**，即在注意力层之间插入循环神经网络（RNN）层（如 Gated DeltaNet, GDN），以提升效率和建模能力。

这类 RNN 层具有**结构性因果性（structurally causal）**，难以像注意力层那样简单地通过移除因果掩码来实现双向化（bidirectionalization），从而阻碍了其直接用于 DLM 的训练。

本文提出并验证了一个关键问题：  
> **能否将这种以 RNN 为主的混合架构成功适配为高效的 DLM？**

### **提出了什么新方法或新思路**

作者提出了 **dQwen3.5** 系列模型，是首个系统性研究如何将**混合架构 AR 模型**转化为 DLM 的工作。

- **核心方法**：仅对 Qwen3.5 中的 **attention layers 进行双向化**，而保留其 **GDN layers 的因果性不变**。
- **创新点**：
  - 首次证明：即使大部分网络堆栈（stack）保持因果性（RNN），仍可通过稀疏的双向注意力注入未来上下文，使模型具备 DLM 所需的任意顺序生成（any-order generation）能力。
  - 提出了一种**轻量级适配策略**，无需修改 RNN 结构本身，即可实现从 AR 到 DLM 的高效迁移。

### **相比现有方法的优势**

| 维度 | 优势 |
|------|------|
| **适配效率** | 在相同训练目标下，dQwen3.5 达到特定训练损失所需的 token 数量约为全注意力控制组的一半（见图5），表明混合架构是更高效的起点。 |
| **下游性能** | 在仅用 50B token 适配的情况下，dQwen3.5-9B 在多个基准上优于使用数百甚至数千亿 token 训练的 DLM（如 Dream-7B、LLaDA-8B）。 |
| **计算成本** | 避免了从头训练 DLM 的高昂开销，也避免了对 RNN 架构进行复杂改造的需求。 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **预训练数据**：来自 NVIDIA 的 Nemotron 预训练发布系列，构建了一个固定混合比例的数据流：
  - **50% 代码**
  - **35% 通用文本**
  - **15% 数学**
- 数据打包为 4096 长度序列，并引入轻微截断和填充增强。

### **实验设置**

- **模型规模**：适配了 Qwen3.5 的四个版本：0.8B、2B、4B 和 9B 参数。
- **控制组**：使用 Qwen3-1.7B（全注意力架构）作为对照，其 trunk 参数量（1.41B）与 dQwen3.5-2B（1.37B）接近，便于公平比较。
- **适配方式**：
  - 仅将 attention layers 改为 bidirectional。
  - GDN layers 保持 causal。
  - 使用 token shifting 对齐 AR 输出头。
  - 引入 mask、padding、BOS 特殊 token。
- **训练配置**：
  - 使用时间重加权的 masked diffusion objective。
  - 学习率随模型大小递减（3e-5 → 1e-5）。
  - batch size: 512 × 4096 tokens。
  - 训练步数：25k 和 50k 步，对应约 50B 和 100B 内容 token。

### **评估指标**

#### **解码行为分析**
- **Local AR-ness (ARL)**：衡量相邻位置是否按左→右顺序解码（随机=0.5，严格 AR=1.0）。
- **Global AR-ness (ARG)**：衡量所有位置对之间的全局顺序趋势。
- 目标：高 ARG（全局左→右趋势）+ 低 ARL（局部可乱序），体现“大体有序、局部灵活”的 DLM 特征。

#### **并行解码能力**
- 固定 NFE（Number of Function Evaluations）预算下的准确率。
- 使用 confidence thresholding 动态决定每步解码数量。
- 报告 HumanEval、MBPP、MATH500、GSM8K、MMLU 上的 pass@1 或准确率。

#### **基线对比模型**
| 类型 | 模型 |
|------|------|
| **DLM 基线** | LLaDA-8B, Dream-7B, Dream-Coder-7B, CoDA |
| **AR 父模型** | Qwen3.5 系列, Qwen2.5 系列 |
| **控制组** | dQwen3-1.7B（全注意力适配） |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ **适配速度显著更快**
- 图5显示，在达到相同训练损失时，**dQwen3.5-2B 所需 token 数仅为 dQwen3-1.7B 控制组的约 45%**（中位数 2.21× 加速）。
- 表明混合架构能更快速吸收 diffusion 目标。

#### ✅ **下游任务表现强劲**
- **表3** 显示：
  - 在 ~1.4B trunk 规模，dQwen3.5-2B（50B token 适配）在 6/7 项任务上超过 CoDA（200B token 适配）。
  - 在 6.5–7B trunk 规模，dQwen3.5-9B（50B token）在 4/7 项任务上优于 Dream-7B（580B）、Dream-Coder-7B（322B）和 LLaDA-8B（2.3T）。

#### ✅ **支持任意顺序与并行解码**
- **图7**：dQwen3.5-9B 的 local AR-ness ≈ 0.636，global AR-ness ≈ 0.967，与其他 DLM（如 LLaDA-8B）处于同一水平，说明其具备典型的 DLM 解码模式。
- **图8 & 图12**：在 increasing NFE speedup 下，dQwen3.5-9B 在 HumanEval 上始终领先其他 DLM，尤其在高速度下优势明显。

#### ✅ **更长适配不一定更好**
- **表2** 显示：
  - 小模型（0.8B）继续训练至 100B token 可提升多数指标。
  - 大模型（9B）进一步训练反而导致 6/7 项指标下降。
- 表明存在一个“最佳适配窗口”，过度训练可能损害性能。

---

## 4. 关键结论和发现

### **主要发现**

1. **混合架构可以成为优秀的 DLM 起点**  
   即使 RNN 层保持因果性，只要 attention 层双向化，就能有效传播未来上下文，支持 any-order 和 parallel decoding。

2. **混合架构适配效率更高**  
   相比全注意力模型，混合模型在更少 token 下即可收敛到相似训练损失，且下游性能更强。

3. **自然语言的因果偏好与 DLM 并不矛盾**  
   实验发现所有 DLM（包括从零训练的 LLaDA）都表现出强烈的 global left-to-right 倾向，说明人类语言本身的因果结构反而是有利归纳偏置。

4. **适配预算应随模型规模调整**  
   更大的模型可能不需要更长的适配过程；50B token 已足够释放其 DLM 潜力。

### **局限性**

1. **架构对比有限**：主要对比集中在 dQwen3.5-2B 与 dQwen3-1.7B，缺乏更多不同比例 RNN/attention 的消融。
2. **依赖特定父模型差异**：两个控制组的 tokenizer 不同，影响 per-token loss 的直接可比性。
3. **未探索 post-training 效果**：仅评估 base DLM，未研究 SFT 或 RLHF 后的表现。
4. **未解释为何小模型受益于更长训练而大模型不然**。

### **未来工作方向**

1. **设计受控的 ablation study**：从零预训练一对完全相同的 hybrid vs full-attention AR 模型，再统一适配为 DLM，以隔离架构影响。
2. **探索最优 RNN/attention 比例**：研究多少比例的双向层足以支撑有效的 DLM 行为。
3. **研究适配机制的本质**：为什么混合架构适配更快？是否与信息瓶颈或梯度传播有关？
4. **扩展到其他 RNN 结构**：如 Mamba、RWKV 等状态空间模型（SSM）是否也可类似适配？

---

> **总结一句话**：  
> dQwen3.5 成功证明了 **“混合因果与非因果结构”** 是一条通往高效、高性能 DLM 的可行路径，挑战了“必须全双向才能做 DLM”的传统认知，为低成本扩散语言建模开辟了新方向。

</details>

---

### 12. [FedeRICo: Federated Region-Influenced Coupling for Traffic Flow Prediction](https://arxiv.org/abs/2609.20026)

**Authors**: Fermin Orozco, Man Luo, Johan Wahlstr\"om  
**Category**: cs.AI  
**Published**: 2026-09-18  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.20026v1  

#### Abstract
Urban traffic forecasting often relies on information distributed across stakeholders who may be unable to share raw data due to privacy or commercial constraints, motivating federated spatial-temporal approaches. In such federated settings, each client observes traffic over a distinct sensor subgra...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《FedeRICo: Federated Region-Influenced Coupling for Traffic Flow Prediction》核心总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在城市交通流量预测中，多个组织（如不同市政部门）通常拥有各自区域的传感器数据，但由于**隐私或商业限制**，无法集中共享原始数据。传统的联邦学习（Federated Learning, FL）方法虽然支持分布式训练，但在以下两方面存在显著缺陷：

1. **参数聚合导致表示稀释**：在异构图结构（heterogeneous graph domains）上直接进行模型参数平均（如 FedAvg），会破坏客户端特有的空间-时间模式，削弱个性化建模能力。
2. **网络分区割裂空间依赖**：将连通的道路网络按客户端划分后，跨边界节点之间的动态传播被切断，导致重要交通扰动（如拥堵、事故）无法跨区传递。

### **提出的新方法与创新思路**
为解决上述问题，本文提出 **FedeRICo**（Federated Region-Influenced Coupling），其核心创新包括两个互补机制：

#### ✅ **1. 双分支预测架构（Dual-Branch Architecture）**
- **Shared Branch（全局分支）**：捕捉所有客户端共有的可迁移预测结构，通过梯度对齐进行协同优化。
- **Local Branch（本地分支）**：保留客户端特异性修正，并接收来自邻近客户端的**边界残差消息**（boundary residual messages），用于恢复跨区空间依赖。
- 最终预测为两者之和：  
  $$
  y^{(m)} = y_{\text{sh}}^{(m)} + y_{\text{loc}}^{(m)}
  $$

#### ✅ **2. 梯度级协作 + 边界感知残差通信**
- **梯度方向对齐（Bi-Gradient Regularisation）**：
  - 不再采用参数平均，而是协调客户端在**梯度空间中的更新方向**。
  - 共享分支梯度鼓励与其他客户端对齐（alignment），而本地分支梯度则避免趋同（diversification），防止个性化信息丢失。
- **边界残差消息传递（Boundary Message Composition）**：
  - 对相邻客户端的边界节点提取**趋势-残差分解**后的瞬态信号（transient spatio-temporal residuals）。
  - 仅传输经过编码的**非周期性短期偏差**，不泄露长期模式或敏感区域特征，兼顾信息增益与隐私保护。

### **相比现有方法的优势**
| 方面 | 传统方法（如 FedAvg, FedGTP） | FedeRICo |
|------|-------------------------------|---------|
| 协作方式 | 参数平均或表征聚合 | 梯度方向对齐 + 残差消息通信 |
| 异构适应性 | 易稀释客户端特性 | 保留个性化建模能力 |
| 跨区依赖建模 | 忽略物理边界连接 | 显式恢复边界动态传播 |
| 隐私保障 | 可能泄露原始轨迹 | 仅传短暂残差信号，抗重构攻击 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
在四个真实世界交通数据集上进行验证：

| 数据集 | 传感器数 | 样本数 | 预测目标 |
|--------|----------|--------|-----------|
| **METR-LA** | 207 | 34,272 | Speed |
| **PEMS-BAY** | 325 | 52,116 | Speed |
| **PEMS03** | 358 | 26,208 | Traffic Flow |
| **PEMS07 (M)** | 228 | 12,672 | Speed |

所有数据以 **5分钟间隔采样**，输入窗口 $T=12$，预测未来 $T'=12$ 步。

### **实验设置**
- **划分策略**：使用 Voronoi 图将路网划分为 $k=4$ 到 $k=20$ 个互斥子图，每个客户端持有独立子图及其数据。
- **训练配置**：
  - 优化器：Adam ($lr=1e^{-3}$)
  - Batch Size：128
  - 联邦轮次：100 global rounds，每轮 1 local epoch
  - 平台：PyTorch + NVIDIA RTX 4090 GPU
- **评估指标**：
  - MAE（Mean Absolute Error）
  - RMSE（Root Mean Squared Error）
  - MAPE（Mean Absolute Percentage Error）

### **基线方法对比**
#### **中心化模型（Centralised）**
- **STDN**：基于时空嵌入的趋势-季节分解模型
- **GWNet**：结合空洞时序卷积与自适应图卷积
- **AGCRN**：学习节点特定嵌入构建动态图

#### **联邦学习基线（Federated）**
- **FedAvg**：标准参数平均框架
- **MFVSTGNN**：多视角联邦图神经网络
- **FedGTP**：基于图协作的跨客户端依赖建模
- **pFedCTP**：自适应参数聚合实现个性化
- **FedDis**：因果解耦框架分离共享与私有表征

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table 2）**
FedeRICo 在所有数据集上均取得最优性能：

| 方法 | METR-LA (MAE) | PEMS-BAY (MAE) | PEMS03 (MAE) | PEMS07 (MAE) |
|------|----------------|------------------|---------------|---------------|
| Centralised GWNet | 3.09 | 1.61 | 14.81 | 2.60 |
| FedAvg | 3.93 | 1.86 | 21.07 | 3.03 |
| FedDis | 3.51 | 1.79 | 16.54 | 2.97 |
| **FedeRICo** | **3.19** | **1.67** | **15.89** | **2.71** |

> 📌 **说明**：FedeRICo 的 MAE 仅比最佳中心化模型（GWNet）高 **3.2%~7.3%**，远优于其他联邦方法。

### **与最强基线 FedDis 的相对提升**
- METR-LA：↓9.1% MAE（3.51 → 3.19）
- PEMS-BAY：↓6.7% MAE（1.79 → 1.67）
- PEMS03：↓3.9% MAE（16.54 → 15.89）
- PEMS07：↓8.8% MAE（2.97 → 2.71）

> 💡 特别是在 METR-LA 上，FedeRICo 甚至**超过中心化 AGCRN 模型**（3.26），表明其有效性不依赖于数据集中化。

### **消融实验结果（Ablation Study）**
（见 Figure 3）

| 变体 | 描述 | METR-LA MAE 影响 |
|------|------|------------------|
| **w/o Coll** | 无联邦协作（纯本地双分支） | ↑从 3.19 → 3.30 |
| **w/o BM** | 移除边界消息 | ↑3.19 → 3.28 |
| **w/o GA** | 替换为参数平均（FedAvg style） | ↑3.19 → 3.49 |
| **w/ OS** | 基于算子相似性的客户选择 | 性能相近，未显著提升 |

> 🔍 结论：**边界消息** 和 **梯度对齐** 是互补且必要的组件；简单增加模型容量不足以替代跨客户端协作。

### **超参数鲁棒性分析（Figure 4）**
- 改变边界消息维度 $d_m$、top-k 上下文数量、梯度控制器强度 $\lambda_{\text{align}}, \delta$ 等，性能波动极小。
- 表明 FedeRICo 设计具有良好的**泛化性和稳定性**，无需精细调参。

### **可扩展性测试（Table 3）**
随着客户端数量从 4 增加到 20：
- FedeRICo 始终保持领先，且性能下降最缓。
- 表明该方法在更细粒度分区下仍有效，具备良好**客户端可扩展性**。

### **计算成本比较（Table 4）**
| 方法 | 参数量（每客户端） | 每轮训练时间 |
|------|--------------------|--------------|
| pFedCTP | 69.0K | 58.6s |
| FedDis | 1.52M | 148.4s |
| **FedeRICo** | **165.9K + 1.3K/encoder** | **21.3s / 40.7s**（预训练/主训练） |

> ⏱️ 尽管引入额外模块，FedeRICo 的**主训练阶段速度最快**，整体效率优于大多数基线。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **梯度空间协作优于参数平均**：在异构图场景下，直接参数聚合会损害客户端个性化表示；而基于梯度方向的对齐机制能更柔和地实现知识迁移。
2. ✅ **边界残差通信可恢复跨区依赖**：通过交换去趋势化的瞬态信号，可在不暴露原始数据的前提下重建部分被分割的空间动态。
3. ✅ **双分支设计实现功能解耦**：共享结构由全局引导，局部修正由边界消息驱动，二者协同提升预测精度。
4. ✅ **高性能不依赖大规模模型或高开销**：FedeRICo 在参数量适中、训练时间可控的情况下实现了SOTA性能。

### **隐私安全性验证**
- 进行“受损接收方重构审计”（compromised-receiver reconstruction audit）：
  - 攻击者试图利用接收到的边界包反推原始流量。
  - 实验显示：仅凭边界消息 + 元数据无法有效重构原始信号（Corr. ≈ 0.2, R² < 0）。
- 结论：所提边界消息机制具备一定**抗逆向工程能力**，符合隐私保护要求。

### **局限性**
1. 当前边界消息仅在**物理相邻客户端间传递**，未考虑远程影响（如全城级事件）。
2. 梯度对齐机制假设所有客户端参与每一轮通信，尚未支持**异步或部分参与**设定。
3. 边界编码器需预先训练并冻结，灵活性受限。

### **未来工作方向**
- 扩展至**异步联邦学习**场景，支持动态客户端加入/退出。
- 引入更强的形式化隐私保障机制（如 Differential Privacy）于边界消息传输过程。
- 探索**多跳边界传播**或图注意力机制以建模远距离依赖。
- 应用于其他时空系统（如空气质量预测、电力负荷调度）。

---

> ✅ **总体评价**：  
> FedeRICo 提出了一种新颖且高效的联邦交通预测框架，通过 **gradient-level collaboration** 与 **boundary-aware residual communication** 的双重机制，在保护隐私的同时显著提升了异构客户端下的预测性能，是 Federated Spatial-Temporal Learning 领域的重要进展。

</details>

---

### 13. [Relational Attention for Data-Efficient Language Modeling](https://arxiv.org/abs/2609.20530)

**Authors**: Adrian Brasoveanu, Ece Takmaz, Jakub Dotla\v{c}il  
**Category**: cs.CL  
**Published**: 2026-09-18  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.20530v1  

#### Abstract
We present Relational BabyLM, a system submission to the BabyLM 2026 challenge that combines two cognitively motivated inductive biases in a single decoder-only Transformer. Architecturally, we replace standard self-attention with a Dual Attention Transformer (DAT), which separates the routing of ob...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Relational Attention for Data-Efficient Language Modeling 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对标准 **Transformer** 在低数据量场景下的两个结构性缺陷：
1. **特征纠缠问题**：标准 self-attention 将对象级（object-level）的“感官”特征与结构/关系信息（structural/relational information）混合在一起，不利于模型对抽象关系进行泛化。
2. **缺乏历史压缩机制**：与 RNN 不同，Transformer 可以无限制地回看所有历史 token，缺乏将序列历史压缩为紧凑信念状态（belief state）的内在压力，这与人类语言处理中的 “Now-or-Never” 瓶颈相悖。

### 提出的新方法与创新
作者提出 **Relational BabyLM**，结合两种认知启发的归纳偏置（inductive biases），提交至 BabyLM 2026 挑战赛：

#### （1）Dual Attention Transformer (DAT) 架构
- **核心思想**：将注意力机制解耦为两条并行流：
  - **Sensory Attention (SA)**：负责路由词项级别的感官特征（如传统 self-attention）。
  - **Relational Attention (RA)**：独立计算接收者与源之间的**显式关系向量**（relation vector），并附加一个标识源的**抽象符号**（abstract symbol）。
- 这种设计受认知科学启发，模仿婴儿能从少量样本中快速学习代数规则（如 ABA vs. ABB）的能力。

#### （2）Next-Latent Prediction (NextLat) 训练目标
- 引入一个辅助的动态模型，训练隐藏状态 $ h_t $ 预测下一个隐藏状态 $ h_{t+1} $，即：
  $$
  h_{t+1} = h_t + \delta([x_{t+1}; h_t])
  $$
- 通过 Smooth L1 和 KL 散度损失监督预测，促使隐藏状态成为对未来预测充分的“信念状态”。
- **关键点**：该辅助模块在推理时被丢弃，不改变原始架构和自回归生成方式。

#### （3）新型符号检索机制：RoPE-based relative symbols
- 提出一种无需学习参数的相对符号生成方法：利用 **Rotary Position Embedding (RoPE)** 直接生成相对位置符号。
- 优势：**零额外参数**，性能媲美需学习的符号库。

---

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **数据效率** | 在仅 10M–100M 单词的极小数据下仍表现出色，优于 GPT-2 baseline。 |
| **结构泛化能力** | 在语法结构任务（如 island effects、主谓一致）上显著提升。 |
| **认知对齐性** | 更好拟合人类阅读时间与 EEG 数据，说明其内部处理更接近人脑。 |
| **参数效率** | RoPE-based 符号机制实现零参数开销，替代需百万级参数的学习符号库。 |

---

## 2. 核心实验方法和设置

### 数据集
- 使用 **BabyLM 2026 Challenge** 提供的两个训练轨道：
  - **strict-small track**: 10 million words
  - **strict track**: 100 million words
- 文本来源包括：CHILDES, OpenSubtitles, Simple Wikipedia, Gutenberg, BNC spoken, Switchboard —— 均为发展心理学支持的儿童导向语料。

### 实验设置
- **模型规模**：
  - strict-small: 12 层，768 维
  - strict: 多种配置，最高达 16 层宽模型（1024 维）
- **优化器**：
  - 对比 AdamW 与自研 **Muon/LambW** 混合优化器（Muon 用于权重矩阵，LambW 用于嵌入等）。
- **上下文长度**：512 tokens（部分消融实验用 264）

### 评估指标
#### Zero-shot 评估套件（无需微调）：
| 指标 | 描述 |
|------|------|
| **BLiMP** | 测试语法最小对立对（minimal pairs），衡量句法结构理解能力 |
| **BLiMP supplement** | 扩展的句法测试集 |
| **EWoK** | 世界知识理解（物理状态、空间关系等） |
| **Entity Tracking** | 实体追踪与状态更新能力 |
| **COMPS** | 概念属性推理与继承 |
| **GlobalPIQA** | 物理常识推理 |
| **Reading Time & EEG Fit** | 与人类眼动、ERP 等神经认知信号的相关性 |
| **Age of Acquisition (AoA)** | 模型习得词汇顺序是否匹配儿童语言发展规律 |

#### 微调任务：
- **(Super)GLUE**：标准 NLP 下游任务集合

---

### 基线方法对比
- **官方 GPT-2 baseline**（98M 参数，AdamW 训练）
- 标准 Transformer（12 heads self-attention）
- 不同变体的 DAT 架构（RA, RCA, DisRCA）
- 不同符号机制（learned relative, RoPE-based, symbolic attention 等）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（严格赛道，100M words）

| 模型 | BLiMP | EWoK | Entity Trk. | COMPS | GLUE* | Overall Avg | 排名 |
|------|-------|------|-------------|--------|--------|--------------|------|
| GPT-2 baseline | 74.73 | 54.37 | 16.91 | 55.85 | 67.75 | 40.73 | — |
| **Best DAT (16L wide)** | **78.94** | **59.54** | **20.47** | **58.69** | **71.43** | **43.92** | **6th / 55** |
| **NLP-task subset** | — | — | — | — | — | — | **3rd / 55** |

> ✅ 最佳模型在 **EWoK 上取得榜单最高分（59.54）**

---

### 与基线对比结果
- 在大多数 benchmark 上**超越 GPT-2 baseline**
- 在 **BLiMP** 上平均高出约 **2.5–4.2 分**
- 在 **EWoK** 上提升显著（+5+ points）
- 在 **(Super)GLUE** 微调任务中，NextLat 模型在 7 项中有 5 项表现更好（如 MultiRC +5.7%, WSC +3.8%）

---

### 消融实验结果

#### （1）Architecture vs. Objective 影响分离
| 因素 | 对结构泛化（BLiMP）影响 | 对认知对齐（Reading）影响 |
|------|--------------------------|----------------------------|
| **Architecture (DAT)** | ⭐⭐⭐ 主导因素（$ \chi^2 = 139.43, p<0.001 $） | 次要 |
| **Objective (NextLat)** | 次要但显著（+0.83 BLiMP pts） | ⭐⭐⭐ 显著提升解释方差（$ t=5.22, p<0.001 $） |

> 🔍 结论：**DAT 决定语法能力上限，NextLat 提升认知合理性**

#### （2）Relational Attention 类型比较（10M 数据）
三种 RA 类型在 10M 数据下性能相当：
- **RCA**（Relational Cross-Attention）：数值略高
- **RA**（Full Relational Attention）与 **DisRCA**：无统计差异
- 原因：简单 RCA 已足够，且无需额外 relation projection 参数

> 📈 但在 100M 数据下，**RA 开始领先**（如 BLiMP 79.30 vs. 78.44），表明其更强的数据扩展潜力。

#### （3）符号检索机制比较（五种子条件 × 5 种随机种子）
| 符号机制 | BLiMP 准确率 | 是否显著优于 baseline？ |
|---------|---------------|------------------------|
| `relative` (learned) | 69.34% | — |
| `relative_rope` (**RoPE-based**) | **69.43%** | ❌ 无显著差异 |
| `positional`, `sinusoidal`, `symbolic` | ~68.9–69.4% | ❌ 无显著差异 |
| `relsymbolic`, `relsymbolic_n4` | 68.33%, 66.32% | ✅ 显著更差 |

> ✅ **RoPE-based relative symbols 性能匹配学习符号库，且零参数成本**

#### （4）SA/RA 头比例扫描（head-ratio sweep）
- 最优比例为 **6SA/6RA**（平衡分配），BLiMP 达峰值 70.33%
- 提交模型采用 **9SA/3RA**，虽牺牲 0.6 BLiMP 分，但换来 **+1.4 补充集增益** 和最佳 COMPS 成绩
- **更高 RA 比例反而降低 BLiMP 和 EWoK 性能**（slope = -1.827, p < 0.001）

> 🧠 启示：感官信息仍是主导，关系流应作为补充而非替代

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **DAT 架构是提升结构泛化的关键**：相比标准 Transformer，在句法任务（尤其是 island effects、subject-verb agreement）上有显著提升。
2. ✅ **NextLat 提升认知对齐性**：模型 surprisal 更好解释人类阅读时间和神经响应，说明其内部表征更具心理现实性。
3. ✅ **RoPE-based relative symbols 是高效替代方案**：无需学习参数即可达到与 learned symbol library 相当性能。
4. ✅ **在小数据下，RCA 与 RA 性能相当**；但在大数据下，**full RA 更具潜力**。
5. ❗ **并非越多关系头越好**：增加 RA 头比例会损害整体性能，最优为平衡或轻微偏向 SA。

---

### 局限性
1. **统计推断保守性不足**：
   - GLMM 模型未建模观察层级随机效应，可能低估标准误。
2. **复现不完整**：
   - 多数实验组仅使用 3 个随机种子，仅符号消融实验完成 5-seed 完整设计。
3. **控制变量不完全**：
   - 提交模型与 baseline 在参数量、优化器、LM head 是否 tied 上均不同，非公平比较。
4. **未探索参数匹配基线**：
   - 缺少与相同 head 数的标准 Transformer 的直接对比。
5. **实现细节差异**：
   - 使用 SwiGLU 而非 GELU、RoPE 替代 learned symbols 等，可能影响与原 DAT 的可比性。
6. **优化器混淆效应**：
   - Muon/LambW 的改进可能是整个 recipe（optimizer + lr + wd）共同作用，无法归因于单一组件。

---

### 未来工作方向
1. **探索更稳定的 full RA 训练策略**：当前 RA 在高维下易发散，需更鲁棒的初始化或训练流程。
2. **扩展到多语言与跨文化场景**：结合 GlobalPIQA 的多语言特性，研究关系注意力在跨语言迁移中的作用。
3. **进一步解耦认知机制**：尝试将 DAT 与更多认知模型（如 ACT-R、predictive coding）结合。
4. **应用于生成任务**：目前评估集中于判别与理解任务，未来可加入 generation quality 评估。
5. **理论分析关系表示的几何结构**：研究 relation vectors 在隐空间中的分布特性及其与形式语义的对应关系。

--- 

> 💡 **总体评价**：本文成功验证了 **relational attention** 在真实语言建模任务中的有效性，特别是在低资源条件下展现出优越的数据效率和结构泛化能力。提出的 **RoPE-based symbol** 机制也为参数高效设计提供了新思路。尽管存在若干实验控制上的局限，但其系统性的消融分析和多维度评估使其成为 BabyLM 挑战中的代表性工作之一。

</details>

---

### 14. [Learning-Based Reconstruction of Optical Properties in Bilayered Media from Single-distance Time-Resolved Reflectance Measurements](https://arxiv.org/abs/2609.19786)

**Authors**: Caterina Amendola, Giulia Maffeis, Lorenzo Buffoni, Lorenzo Chicchi, Francesco Coghi, Duccio Fanelli, Raffaele Marino, Fabrizio Martelli, Riccardo Paoli, Lorenzo Pattelli, Lorenzo Spinelli  
**Category**: cs.LG  
**Published**: 2026-09-18  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.19786v1  

#### Abstract
The inverse problem of reconstructing optical properties, specifically absorption and scattering coefficients, in layered biological media from time-domain reflectance measurements remains a significant challenge for traditional analytical models. Inverse solvers based on the diffusion equation ofte...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Learning-Based Reconstruction of Optical Properties in Bilayered Media from Single-distance Time-Resolved Reflectance Measurements

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**双层生物介质中光学性质（吸收系数 $ \mu_a $ 和约化散射系数 $ \mu_s' $）的反演重建问题**，尤其是在**单距离时间分辨反射测量**（single-distance time-resolved reflectance）条件下，传统基于扩散方程（Diffusion Equation, DE）的模型驱动反演方法存在以下挑战：
- 在结构异质性下精度下降；
- 对表层吸收和深层散射的估计准确性差；
- 受限于DE近似带来的系统误差；
- 需要多距离测量或多先验信息才能获得较好结果。

### 提出的新方法与新思路
提出了一种**基于机器学习（Machine Learning, ML）的端到端反演框架**，其核心创新包括：

- **完全基于精确蒙特卡洛模拟生成的合成数据集进行训练**：避免了扩散近似的理论偏差，提升了前向建模的物理保真度。
- **引入 Spectral Autoencoder (SPAE)** 进行无监督特征提取与**内在维度估计**：自动识别DTOF数据的有效自由度数量（实验中为4），无需预设层数或参数个数。
- **构建两阶段学习管道**：
  1. 使用SPAE从DTOF曲线中提取低维潜在表示；
  2. 将几何信息（source-detector distance $ p $、上层厚度 $ L $）与潜在变量拼接后输入多头MLP分类器，实现对四个光学参数的联合预测。
- **仅使用单距离测量数据**，代表最信息受限的现实场景，验证方法在极限条件下的鲁棒性。

### 相比现有方法的优势
| 维度 | 优势说明 |
|------|----------|
| **准确性** | ML方法在绝大多数参数上的重建准确率显著高于基于DE+Levenberg-Marquardt的传统方法，尤其在深层吸收 $ \mu_{a,t} $ 和表层散射 $ \mu_{s,r} $ 上提升明显。 |
| **计算效率** | ML推理速度达**毫秒级**，而传统方法需数天完成全数据集反演，加速数个数量级。 |
| **无需良好初值** | 传统方法严重依赖初始猜测，而ML方法不依赖初始化，在无先验情况下仍保持高精度。 |
| **自适应维度感知** | SPAE能自动识别DTOF数据的内在维度（≈4），可用于指导模型复杂度选择和结构推断。 |

---

## 2. 核心实验方法和设置

### 数据集
- **来源**：通过GPU加速的**Monte Carlo (MCX)** 软件包生成的**合成数据集**。
- **规模**：共 **399,300 条 DTOF 曲线**。
- **覆盖范围广泛**：
  - 上层厚度 $ L $: 2–40 mm（11个离散值）
  - 源-探测距离 $ p $: 10, 20, 30 mm
  - 吸收系数 $ \mu_a $: $ 10^{-4} \sim 0.5\ \text{mm}^{-1} $
  - 约化散射系数 $ \mu_s' $: $ 0.5 \sim 1.5\ \text{mm}^{-1} $
- **时间分辨率**：10 ps/bin，共1000 bins（总时窗10 ns）
- **IRF假设**：Dirac delta函数，排除仪器响应影响。

### 实验设置
- **任务形式**：将连续参数离散化为类别，转化为**多标签分类问题**。
  - $ \mu_a $ 分为10类，$ \mu_s' $ 分为11类。
- **训练/测试划分**：80%/20%，随机分割。
- **评估指标**：
  - **Top-1 Accuracy**：预测类别是否落在真实值与其邻近值中点构成的“正确区间”内。
  - **Mean Class Shift (MCS)**：衡量预测偏离对角线的程度，越小越好。

### 基线方法对比
- **Analytical Model-based Reconstruction**：
  - 前向模型：基于DE的双层解析解 [37–39]
  - 反演算法：Levenberg-Marquardt非线性优化
  - 初始值策略：使用均匀介质模型拟合得到的等效参数作为初值（模拟真实实验条件）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Test Set）

| 参数 | ML 准确率范围 | Analytical 准确率范围 | 提升幅度 |
|------|----------------|------------------------|---------|
| $ \mu_{a,r} $（表层吸收） | 89.3% – 96.4% | 10.6% – 96.9% | 平均↑5.8% |
| $ \mu_{a,t} $（深层吸收） | 89.2% – 95.8% | 10.5% – 92.3% | 平均↑8.5% |
| $ \mu_{s,r} $（表层散射） | 88.7% – 96.9% | 20.7% – 84.0% | 平均↑27.6% |
| $ \mu_{s,t} $（深层散射） | ≤34% | ≤30.4% | 两者均不可靠 |

> 注：所有准确率随 $ p $ 增大而提高；对于固定 $ p $，表层参数随 $ L $ 增加更易恢复，深层则相反。

### 与基线方法对比结果
- **总体表现**：
  - ML在三个参数（除 $ \mu_{s,t} $ 外）上全面超越传统方法。
  - 特别是在低吸收区域（$ \mu_a < 10^{-3}\ \text{mm}^{-1} $）和高散射区域，ML仍保持较高准确率。
- **误差分析（Mean Class Shift）**：
  - $ \mu_{a,r} $: MCS 从 1.2 → 0.85
  - $ \mu_{a,t} $: MCS 从 1.56 → 0.93
  - $ \mu_{s,r} $: MCS 从 0.83 → 0.28
  - $ \mu_{s,t} $: MCS 从 3.53 → 3.39（改善有限）
- **计算成本**：
  - Analytical 方法：约 **6天**（并行40核CPU）
  - ML 方法：训练约 **15分钟**（单块NVIDIA RTX-A5500），推理每次 **几毫秒**

### 消融实验与补充验证（Appendix B）
- **初始条件敏感性分析**：
  - 当Analytical方法以**真实参数作为初值**（理想情况），其性能可超过ML。
  - 但在实际应用场景中（无先验知识），ML始终优于传统方法。
  - 表明ML更具**实用性和鲁棒性**。
- **SPAE维度识别能力验证（Appendix A）**：
  - 在简单均匀介质中，SPAE自动识别出1个主成分；
  - 在双层+源距条件下，识别出5个相关维度（4个光学参数 + 1个几何参数），证明其具备**自组织建模能力**。

---

## 4. 关键结论和发现

### 主要发现
1. **ML方法显著优于传统模型驱动反演方法**，在准确性、鲁棒性和速度方面均有压倒性优势。
2. **深层散射系数 $ \mu_{s,t} $ 的重建极不可靠**，即使采用ML也无法解决——这表明问题根源可能不在算法本身，而是**单距离时间分辨测量对深层散射缺乏灵敏度**这一根本物理限制。
3. **SPAE能够有效估计DTOF数据的内在维度为4**，与双层介质中的四个独立光学参数一致，展示了其在**模型选择与结构识别方面的潜力**。
4. **该框架可在无任何关于层数的先验信息下运行**，仅需输入上层厚度 $ L $ 和源距 $ p $，具有良好的泛化前景。

### 方法的局限性
- **无法可靠重建深层散射 $ \mu_{s,t} $**：这是当前单距离TD测量的根本瓶颈，非算法所能克服。
- **依赖高质量合成数据**：训练依赖大量MC模拟，虽物理准确但计算开销大。
- **尚未考虑真实噪声与仪器响应函数**：目前使用Dirac delta IRF，未加入真实系统的展宽效应。
- **厚度 $ L $ 被视为已知量**：现实中 $ L $ 存在测量不确定性，可能影响性能。

### 未来工作方向
1. **扩展至多距离联合重建**（multi-distance reconstruction）：利用不同 $ p $ 提供的深度敏感性互补信息，有望突破 $ \mu_{s,t} $ 的重建瓶颈。
2. **引入更真实的实验条件**：
   - 加入有限IRF、探测器噪声、采样率限制等；
   - 引入 $ L $ 的不确定性建模。
3. **推广至更多层数和复杂几何结构**：探索ML在N层介质中的适用性。
4. **发展混合方法（Hybrid Approach）**：
   - 结合物理模型的可解释性与ML的速度/灵活性；
   - 如用ML提供初值给传统优化器，或构建物理约束的神经网络。
5. **迁移到 in vivo 数据**：最终目标是应用于人体组织（如皮肤、脑皮层）的实时监测。

---

> ✅ **总结一句话**：  
> 本文提出一种基于ML的双层介质光学性质重建新范式，利用高保真MC数据和SPAE特征提取，在单距离TD测量下实现了比传统方法更快、更准、更鲁棒的反演，并揭示了深层散射难以重建的根本物理限制，为未来开发临床可用的快速光学成像系统提供了重要基础。

</details>

---

### 15. [QoS-Aware Federated Learning for Multimodal In-Cabin Interaction in Smart Vehicles](https://arxiv.org/abs/2609.20123)

**Authors**: Baran Can G\"ul, Mert Nak{\i}p, Nasser Jazdi, Michael Weyrich  
**Category**: cs.LG  
**Published**: 2026-09-18  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.20123v1  

#### Abstract
Modern smart vehicles leverage multimodal sensors, ranging from high-bandwidth vision systems to low-rate physiological monitors, to provide personalized in-cabin services. However, integrating high-fidelity multimodal fusion with collaborative training is often hindered by the heterogeneous and tim...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**QoS-Aware Federated Learning for Multimodal In-Cabin Interaction in Smart Vehicles**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代智能车辆配备多模态传感器（如摄像头、生理信号监测器等），用于实现个性化座舱交互服务。然而，在车载网络中部署标准的 **Federated Learning (FL)** 面临以下三大挑战：
- **Safety-Critical Timing Violations**：传统 FL 忽略推理延迟要求（如疲劳检测需 <100ms），导致资源受限 ECU 上平均延迟高达 1.598s。
- **Energy Budget Exhaustion**：高保真多模态融合消耗大量能量，影响核心驾驶功能供电。
- **Personalization-Resource 冲突**：现有个性化 FL 方法（如 FedPer、APFL）提升精度但增加计算负担，违背 QoS 要求。

这些问题源于现有 FL 方法将模型收敛作为唯一目标，忽视了车载环境中动态变化的 **Quality of Service (QoS)** 约束（带宽、延迟、能耗）。

---

### 🚀 提出的新方法：**FedQoS**
本文提出 **FedQoS** ——一种异步、事件触发式的联邦学习框架，通过**双阶段门控机制（Two-Phase Gating Mechanism）** 解耦本地训练与全局通信，联合优化个性化性能与系统级 QoS 约束。

#### 主要创新点：
1. **Resource-Aware Training Gate（资源感知训练门）**
   - 只有当本地数据缓存满足最小批量 $B_{\text{min}}$ 且剩余能量 $E(t_k) > E_{\text{safe}}$ 时才启动本地训练。
   - 防止 ML 任务干扰车辆安全运行和续航能力。

2. **Staleness-Aware Proximal Objective（陈旧性感知正则项）**
   - 引入时间衰减的 proximal 正则系数 $\mu(t_k) = \mu_0 \cdot e^{-\alpha(t_k - T_i)}$，动态调整客户端对全局模型的一致性强度。
   - 当全局模型较新时强制一致性；当过期时允许客户端优先本地个性化学习。

3. **QoS-Aware Transmission Policy（QoS 感知传输策略）**
   - 定义 **Transmission Efficiency Score**：
     $$
     Z(t_k) = w_m U(t_k) - w_c C(t_k) - w_e \epsilon(t_k)
     $$
     其中包含信息效用（model novelty）、延迟成本和能量影响三项。
   - 仅当 $Z(t_k) \geq \delta$ 时上传更新，否则缓存等待更优时机。

4. **Asynchronous Server Aggregation with Staleness Weighting**
   - 服务器采用基于陈旧度的加权聚合：权重随 $\Delta t = t - T_i$ 增大而衰减（使用 $(1+\Delta t)^{-3}$ 函数），稳定异步收敛。

---

### 🔍 相比现有方法的优势
| 维度 | FedAvg / 标准 FL | FedQoS |
|------|------------------|--------|
| 同步性 | 同步轮次（rigid rounds） | 异步事件驱动 |
| QoS 支持 | 无显式支持 | 显式建模并控制 |
| 能耗管理 | 忽视能量预算 | 动态能量门限防止耗尽 |
| 通信效率 | 固定频率上传 | 按效用-成本决策是否上传 |
| 个性化适应 | 固定正则强度 | 时间衰减调节一致性 |

> ✅ **核心优势**：在保证接近 FedAvg 的个性化准确率的同时，显著降低通信开销、延迟和能耗，适用于真实车载场景。

---

## 2. 核心实验方法和设置

### 📊 数据集
- 使用一个真实的**多模态驾驶员状态监测数据集**，包含：
  - **Vehicle Telemetry**：车速、转向角等 14 维特征
  - **Physiological Signals**：PPG（光电容积脉搏波）、EEG（脑电图）共 7 特征
  - **Cabin-Facing Video**：面部视频用于情绪/疲劳识别
- 包含六种驾驶状态标签：normal, distracted, alerted, stressed, relaxed, drowsy
- 多参与者数据，按 subject-level 时间划分训练/测试集，模拟 non-IID 分布

---

### ⚙️ 实验设置
- **Client 数量**：$N=10$ 辆虚拟智能车
- **Training Rounds**：20 轮
- **Local Epochs**：每轮 3 epoch
- **Model Architecture**（统一架构以公平比较）：
  - **Telemetry Encoder**：Stacked BiLSTM (64→32)
  - **Physio Encoder**：LSTM (128) + early fusion
  - **Video Encoder**：4-layer 3D-CNN (3→32→64→128→128)
  - **Fusion Module**：8-head Cross-Modal Self-Attention
  - **Classifier Head**：MLP (512→256→6 classes)

- **Server 参数**：
  - Staleness weight decay: $\beta = 0.5$
  - Global learning rate: $\alpha_{\text{global}} = 0.1$

- **QoS 模拟环境**：
  - 动态网络条件（带宽、可靠性、延迟）
  - 客户端总能量预算：1000J
  - 通信代价依赖 payload 大小与瞬时信道状态

---

### 🎯 评估指标
| 类别 | 指标 |
|------|------|
| **Accuracy & Personalization** | Average Test Accuracy, F1-Score |
| **Communication Efficiency** | Total Data Transmitted, Wasted Transmissions |
| **Energy Efficiency** | Total Energy Consumed |
| **Latency & Bandwidth** | Avg. Latency Cost, Avg. Bandwidth, Channel Reliability |
| **Convergence Behavior** | Accuracy vs. Communication Round |

---

### 🆚 基线方法对比
- **FedAvg**：经典同步联邦平均算法（McMahan et al., AISTATS 2017）
- 所有方法使用相同模型结构和训练配置，确保可比性

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Figure 2 和 3）

| 指标 | FedQoS | FedAvg | 提升/节省 |
|------|--------|--------|----------|
| **Total Data Transmitted** | 469.7 MB | 2013.1 MB | ↓ **76.7%** |
| **Wasted Transmissions** | 0.0 MB | 335.5 MB | ✅ **完全消除** |
| **Total Energy Consumed** | 892.1 J | 1000.0 J | ↓ **10.8%** |
| **Avg. Latency Cost** | 1.182 s | 1.598 s | ↓ **26.0%** |
| **Avg. Bandwidth Utilization** | 70.9 Mbps | 55.7 Mbps | ↑ **27.2%** |
| **Channel Reliability** | 0.8854 | 0.8730 | ↑ **1.4%** |
| **Peak Test Accuracy** | ~85% | ~90% | ↓ ~5% |
| **F1-Score** | 略低于 FedAvg | 略高 | 差距较小 |

> 💡 注：尽管 FedQoS 在最终精度上略有下降（约 5%），但在所有 QoS 指标上均取得显著改进。

---

### 🔬 关键观察与分析
- **通信体积大幅减少**：得益于双阶段门控，76.7% 的潜在更新被抑制，尤其是低效或高风险传输。
- **零浪费传输**：FedQoS 在发送前判断信道质量与能量状态，避免无效传输；而 FedAvg 发送后丢包造成资源浪费。
- **更高的有效带宽利用率**：选择信道质量好的客户端上传，提升了实际吞吐量。
- **延迟从 1.598s → 1.182s**：已低于多数安全阈值（如 1.2s），更适合实时应用。
- **能量节省 10.8%**：对于电动车而言，意味着更多余量供给动力系统或其他关键功能。
- **精度损失可控**：仅牺牲约 5% 的准确率换取巨大的系统效益，在安全敏感场景下是合理折衷。

---

### ❌ 消融实验（未明确列出，但从设计可推断）
虽然文中未提供正式消融研究表格，但从机制设计可以推测以下组件的作用：
- 若移除 **training gate** → 更多训练触发 → 能耗上升、可能违反安全阈值
- 若移除 **transmission gate** → 回归固定上传模式 → 浪费传输重现、延迟升高
- 若使用固定 $\mu$ 而非 time-decaying → 无法自适应连接中断，个性化能力下降

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **QoS 与 Personalization 是耦合优化问题**，不能孤立处理。FedQoS 成功将二者纳入统一框架。
2. **事件触发机制优于固定周期同步**：在动态车载环境中，reactive learning 更符合现实约束。
3. **Decoupling Computation and Communication 是关键**：允许车辆“边学边等”，提高资源利用效率。
4. **轻微精度损失换来巨大系统收益是值得的**：尤其在安全攸关系统中，可靠性和可持续性优先于极致精度。

---

### ⚠️ 方法的局限性
1. **门控阈值需要调参**：如 $B_{\text{min}}, E_{\text{safe}}, \delta$ 等需根据具体硬件平台设定，缺乏自动化调节机制。
2. **未考虑极端网络分区**：若长期无连接，本地模型可能过度偏离全局趋势。
3. **缺乏跨模态调度机制**：当前仍训练完整模型，未来可探索 per-modality selective update。
4. **仿真环境简化**：尚未集成高保真 vehicular mobility model 或复杂无线传播模型。

---

### 🔮 未来工作方向（作者明确提出）
1. **Per-Modality Online Learning Strategies**
   - 动态选择重要模态子网络进行更新，进一步降低开销。
2. **High-Fidelity Networking Simulation**
   - 结合真实 vehicular mobility 和 radio propagation 模型验证鲁棒性。
3. **Adaptive Threshold Tuning**
   - 利用 RL 或 control theory 自动调整门控参数。
4. **Integration with SDV 架构**
   - 将 FedQoS 集成至 Software-Defined Vehicle 平台，支持 OTA 升级与弹性资源配置。

---

## 总结

✅ **FedQoS 是首个专为智能汽车多模态座舱交互设计的 QoS-Aware FL 框架**，其核心思想是：
> “**不是每次都能学，也不是每次都要传**”

通过 **双阶段门控 + 陈旧性感知优化**，实现了：
- 接近 FedAvg 的个性化性能
- 显著降低通信（↓76.7%）、能耗（↓10.8%）、延迟（↓26.0%）
- 完全消除无效传输
- 更好地适配车载系统的资源波动与安全约束

📌 **适用场景**：适用于带有多模态传感、资源受限、连接不稳定的真实智能车辆系统，是迈向实用化 vehicular FL 的重要一步。

</details>

---

### 16. [Self Improvement via Fast Tree-search](https://arxiv.org/abs/2609.19526)

**Authors**: Xinghong Fu, Aravinth Kulanthaivelu, Yutaro Yamada  
**Category**: cs.AI  
**Published**: 2026-09-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.19526v1  

#### Abstract
Coding agents can recursively modify their own implementations, forming a loop of self-improvement. While prior work shows this can boost performance on coding benchmarks, existing approaches are costly and compute-intensive. We introduce a simple, sample-efficient self-improvement framework that si...

---

### 17. [When2Think: Learning Difficulty-Aware Length Control for Efficient Hybrid Reasoning Models](https://arxiv.org/abs/2609.19671)

**Authors**: Jaejun Shim, HyunJin Kim, Young Jin Kim, JinYeong Bak  
**Category**: cs.AI  
**Published**: 2026-09-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.19671v1  

#### Abstract
Large Reasoning Models (LRMs) achieve strong performance on complex tasks but exhibit systematic inefficiency: they often overthink easy problems and underthink hard ones. Existing approaches based on uniform length penalties or rigid routing incur an efficiency tax, trading reduced computation on e...

---

### 18. [From "Who Is This User?" to "What Does This Purchase Mean?": A Deployed Pipeline for Semantic User Profiling at Bank Scale](https://arxiv.org/abs/2609.19928)

**Authors**: Ryota Mitsuhashi, Tetsuro Morimura, Hirotake Ito  
**Category**: cs.AI  
**Published**: 2026-09-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.19928v1  

#### Abstract
Per-user LLM inference on transaction histories binds the inference budget linearly to user count, which becomes prohibitive at applied scale. We re-cast attribute inference from per-user to per-transaction-pattern. The pipeline runs in three phases: Resolve abstracts item names with optional web gr...

---

### 19. [AI-Driven Real-Time Relay Optimisation in Smart Urban NR-V2X Networks via Learning-to-Optimise Graph Neural Networks](https://arxiv.org/abs/2609.20271)

**Authors**: Giambattista Amati, Federica Mangiatordi, Emiliano Pallotti, Simone Angelini  
**Category**: cs.AI  
**Published**: 2026-09-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.20271v1  

#### Abstract
Reliable and low-latency communication is a fundamental requirement for smart city services and Industry 4.0 applications enabled by NR-V2X networks. However, limited Road-Side Unit (RSU) deployment and complex urban propagation conditions often prevent Connected and Automated Vehicles (CAVs) from m...

---

### 20. [JEPA-WAM: Connecting Generated Visual Instructions to World Action Models through JEPA Latent Representations](https://arxiv.org/abs/2609.20277)

**Authors**: Tianbin Liu, Jian Zhu, Taiyi Su, Jianjun Zhang, Chong Ma, Zitai Huang, Yi Xu  
**Category**: cs.AI  
**Published**: 2026-09-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.20277v1  

#### Abstract
World Action Models (WAMs) have demonstrated strong robotic manipulation capabilities by augmenting pretrained video generative models with action experts. However, current WAMs still show limited instruction-following ability when conditioned solely on text instructions. We argue that this limitati...

---

### 21. [On-Demand Attention: Language Models Know When to Recall](https://arxiv.org/abs/2609.20734)

**Authors**: Haibo Feng, Ruiqi Liang, Hanyang Peng, Shiqi Yu  
**Category**: cs.CL  
**Published**: 2026-09-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.20734v1  

#### Abstract
Reasoning and agentic workloads increasingly demand efficient long-context inference. Yet full-attention decoding reads the growing history at every step, regardless of its benefit to the next prediction. We show that a pretrained model's decoding states already contain information predictive of thi...

---

### 22. [Radio-Frequency Convolutional Neural Networks](https://arxiv.org/abs/2609.19279)

**Authors**: Zhihui Gao, Shi-Yuan Ma, Yiran Chen, Dirk Englund, Tingjun Chen  
**Category**: cs.LG  
**Published**: 2026-09-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.19279v1  

#### Abstract
Running artificial intelligence (AI) models directly on edge devices such as smartphones, wearables, and drones offers low latency, pervasive scalability, and data privacy, but these devices rarely carry the computing capability that modern neural networks demand. Edge accelerators have been develop...

---

### 23. [Digital Twins for Opinion Dynamics: A Generative LLM Framework for Social Networks](https://arxiv.org/abs/2609.19913)

**Authors**: Omran Berjawi, Giuseppe Fenza, Rida Khatoun, Sherali Zeadally  
**Category**: cs.LG  
**Published**: 2026-09-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.19913v1  

#### Abstract
The study of opinion dynamics in social networks is one of the key challenges in computational social science with direct relevance to understanding political polarization, misinformation, and health responses. Current approaches focus on simplified mathematical models that ignore linguistic and con...

---

### 24. [Dual-Axis Policy Optimization for LLM Agents: Bayesian Feedback Attribution and Trajectory Mass Normalization](https://arxiv.org/abs/2609.19830)

**Authors**: Yingxuan Zhuang, Binhe Yu, Jingxiao Yang, Ruopei Sun, Ziting Li, Cheng Tan, Xuhong Zhang, Jianwei Yin, Jintao Chen  
**Category**: cs.AI  
**Published**: 2026-09-18  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.19830v1  

#### Abstract
Reinforcement learning for LLM agents involves two distinct optimization di- mensions: how environment feedback is exploited within a trajectory, and how complete trajectories are aggregated across a batch. We formulate these dimen- sions as Intra-Trajectory Feedback Attribution and Inter-Trajectory...

---

### 25. [TRACE: Accountable Agentic Retrieval for Source Discovery in Digital Archives](https://arxiv.org/abs/2609.19897)

**Authors**: Donghan Bian (ENC, LRE), Marie Puren (LRE, ENC), Florian Cafiero (LRE, ENC)  
**Category**: cs.AI  
**Published**: 2026-09-18  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.19897v1  

#### Abstract
Historical archives pose a difficult retrieval problem for retrievalaugmented generation systems: documents are OCR-degraded, heterogeneous across genres and sources, and require strong source traceability for scholarly and institutional use. We introduce TRACE, a training-free agentic retrieval fra...

---

### 26. [Solving Minimum Span Antibandwidth and Cyclic Antibandwidth Labeling Problems](https://arxiv.org/abs/2609.20091)

**Authors**: Hieu Truong Xuan, Khanh To Van  
**Category**: cs.AI  
**Published**: 2026-09-18  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.20091v1  

#### Abstract
The Antibandwidth and Cyclic Antibandwidth problems are NP-hard graph labeling problems that aim to maximize the minimum (cyclic) distance between labels assigned to adjacent vertices. Extensive research on these problems has resulted in a variety of mathematical formulations and computational appro...

---

### 27. [JointMatch: A Unified Heterogeneous Graph Neural Solver for Large-Scale Ride-Sharing Matching](https://arxiv.org/abs/2609.20200)

**Authors**: Kun Zhao, Xu Chen  
**Category**: cs.AI  
**Published**: 2026-09-18  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.20200v1  

#### Abstract
Ride-sharing platforms must continuously decide which open requests to bundle into shared trips and which idle vehicles should serve them. The dominant academic approach decomposes this into two sequential matching problems -- request pairing first, then vehicle assignment -- and applies a separate ...

---

### 28. [UniPolicy: Unified Objective-Specific Policies for Generative Search Advertising](https://arxiv.org/abs/2609.20630)

**Authors**: Kun Yao, Yuhang Zhou, Yichi Zhang, Zeliang Tong, Shengri Xue, Haitao Wang, Siyu Lu, Qianlong Xie, Xingxing Wang  
**Category**: cs.CL  
**Published**: 2026-09-18  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.20630v1  

#### Abstract
Search advertising connects user intent with commercial content and plays a critical role in platform monetization. Recent systems typically align pretrained generative models with a single business reward, such as eCPM, or use naive reward fusion for preliminary multi-objective alignment. However, ...

---

### 29. [Bayesian Optimization with Rich Auxiliary Information via LLMs](https://arxiv.org/abs/2609.19437)

**Authors**: Tejus Gupta, Efe Mert Karag\"ozl\"u, Rohit Sonker, Barnab\'as P\'oczos, Jeff Schnieder  
**Category**: cs.LG  
**Published**: 2026-09-18  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.19437v1  

#### Abstract
Bayesian Optimization (BO) is widely used for optimizing expensive black-box functions, yet many real-world optimization problems contain substantially richer information than function evaluations alone. Examples include training curves in hyperparameter optimization, expert notes and images in scie...

---

### 30. [OceanMoE: Structured Conditional Sparse Computation for Long-Horizon Multivariate Ocean Forecasting](https://arxiv.org/abs/2609.19768)

**Authors**: Yishun Zhu, Jian Wang  
**Category**: cs.LG  
**Published**: 2026-09-18  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.19768v1  

#### Abstract
Multivariate ocean forecasting must exploit shared evolution in a coupled ocean system while adapting to the heterogeneous statistical and dynamical characteristics of different prediction variables and locations. Fully shared models may lack the flexibility to handle this heterogeneity, whereas ful...

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

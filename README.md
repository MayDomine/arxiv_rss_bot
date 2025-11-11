# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2025-11-11 12:54:16 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [FlashMoE: Fast Distributed MoE in a Single Kernel](https://arxiv.org/abs/2506.04667)

**Authors**: Osayamen Jonathan Aimuyo, Byungsoo Oh, Rachee Singh  
**Category**: cs.DC  
**Published**: 2025-11-11  
**Score**: 15.0  
**Type**: replace  
**ArXiv ID**: 2506.04667v3  

The computational sparsity of Mixture-of-Experts (MoE) models enables sub-linear growth in compute cost as model size increases, thus offering a scalable path to training massive neural networks. However, existing implementations suffer from low GPU utilization, significant latency overhead, and a f...

---

### 2. [Data-Centric Elastic Pipeline Parallelism for Efficient Long-Context LLM Training](https://arxiv.org/abs/2509.21275)

**Authors**: Shiju Wang, Yujie Wang, Ao Sun, Fangcheng Fu, Zijian Zhu, Bin Cui, Xu Han, Kaisheng Ma  
**Category**: cs.DC  
**Published**: 2025-11-11  
**Score**: 12.5  
**Type**: replace  
**ArXiv ID**: 2509.21275v2  

Long context training is crucial for LLM's context extension. Existing schemes, such as sequence parallelism, incur substantial communication overhead. Pipeline parallelism (PP) reduces this cost, but its effectiveness hinges on partitioning granularity. Batch-level PP dividing input samples exhibit...

---

### 3. [Accelerating LLM Inference Throughput via Asynchronous KV Cache Prefetching](https://arxiv.org/abs/2504.06319)

**Authors**: Yanhao Dong, Yubo Miao, Weinan Li, Xiao Zheng, Chao Wang, Jiesheng Wu, Feng Lyu  
**Category**: cs.LG  
**Published**: 2025-11-11  
**Score**: 11.5  
**Type**: replace  
**ArXiv ID**: 2504.06319v2  

Large Language Models (LLMs) exhibit pronounced memory-bound characteristics during inference due to High Bandwidth Memory (HBM) bandwidth constraints. In this paper, we propose an L2 Cache-oriented asynchronous KV Cache prefetching method to break through the memory bandwidth bottleneck in LLM infe...

---

### 4. [MoSKA: Mixture of Shared KV Attention for Efficient Long-Sequence LLM Inference](https://arxiv.org/abs/2511.06010)

**Authors**: Myunghyun Rhee, Sookyung Choi, Euiseok Kim, Joonseop Sim, Youngpyo Joo, Hoshik Kim  
**Category**: cs.DC  
**Published**: 2025-11-11  
**Score**: 11.0  
**Type**: cross  
**ArXiv ID**: 2511.06010v1  

The escalating context length in Large Language Models (LLMs) creates a severe performance bottleneck around the Key-Value (KV) cache, whose memory-bound nature leads to significant GPU under-utilization. This paper introduces Mixture of Shared KV Attention (MoSKA), an architecture that addresses th...

---

### 5. [VeriLLM: A Lightweight Framework for Publicly Verifiable Decentralized Inference](https://arxiv.org/abs/2509.24257)

**Authors**: Ke Wang, Zishuo Zhao, Xinyuan Song, Bill Shi, Libin Xia, Chris Tong, Lynn Ai, Felix Qu, Eric Yang  
**Category**: cs.LG  
**Published**: 2025-11-11  
**Score**: 10.5  
**Type**: replace-cross  
**ArXiv ID**: 2509.24257v3  

Decentralized inference provides a scalable and resilient paradigm for serving large language models (LLMs), enabling distributed resource utilization and reducing reliance on centralized providers. However, in a permissionless environment without trusted nodes, ensuring the correctness of model out...

---

### 6. [MOSS: Efficient and Accurate FP8 LLM Training with Microscaling and Automatic Scaling](https://arxiv.org/abs/2511.05811)

**Authors**: Yu Zhang, Hui-Ling Zhen, Mingxuan Yuan, Bei Yu  
**Category**: cs.LG  
**Published**: 2025-11-11  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2511.05811v1  

Training large language models with FP8 formats offers significant efficiency gains. However, the reduced numerical precision of FP8 poses challenges for stable and accurate training. Current frameworks preserve training performance using mixed-granularity quantization, i.e., applying per-group quan...

---

### 7. [Precision-Scalable Microscaling Datapaths with Optimized Reduction Tree for Efficient NPU Integration](https://arxiv.org/abs/2511.06313)

**Authors**: Stef Cuyckens, Xiaoling Yi, Robin Geens, Joren Dumoulin, Martin Wiesner, Chao Fang, Marian Verhelst  
**Category**: cs.LG  
**Published**: 2025-11-11  
**Score**: 10.0  
**Type**: cross  
**ArXiv ID**: 2511.06313v1  

Emerging continual learning applications necessitate next-generation neural processing unit (NPU) platforms to support both training and inference operations. The promising Microscaling (MX) standard enables narrow bit-widths for inference and large dynamic ranges for training. However, existing MX ...

---

### 8. [StreamDiffusionV2: A Streaming System for Dynamic and Interactive Video Generation](https://arxiv.org/abs/2511.07399)

**Authors**: Tianrui Feng, Zhi Li, Shuo Yang, Haocheng Xi, Muyang Li, Xiuyu Li, Lvmin Zhang, Keting Yang, Kelly Peng, Song Han, Maneesh Agrawala, Kurt Keutzer, Akio Kodaira, Chenfeng Xu  
**Category**: cs.LG  
**Published**: 2025-11-11  
**Score**: 10.0  
**Type**: cross  
**ArXiv ID**: 2511.07399v1  

Generative models are reshaping the live-streaming industry by redefining how content is created, styled, and delivered. Previous image-based streaming diffusion models have powered efficient and creative live streaming products but have hit limits on temporal consistency due to the foundation of im...

---

### 9. [FedSparQ: Adaptive Sparse Quantization with Error Feedback for Robust & Efficient Federated Learning](https://arxiv.org/abs/2511.05591)

**Authors**: Chaimaa Medjadji, Sadi Alawadi, Feras M. Awaysheh, Guilain Leduc, Sylvain Kubler, Yves Le Traon  
**Category**: cs.LG  
**Published**: 2025-11-11  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2511.05591v1  

Federated Learning (FL) enables collaborative model training across decentralized clients while preserving data privacy by keeping raw data local. However, FL suffers from significant communication overhead due to the frequent exchange of high-dimensional model updates over constrained networks. In ...

---

### 10. [Every Activation Boosted: Scaling General Reasoner to 1 Trillion Open Language Foundation](https://arxiv.org/abs/2510.22115)

**Authors**: Ling Team, Ang Li, Ben Liu, Binbin Hu, Bing Li, Bingwei Zeng, Borui Ye, Caizhi Tang, Changxin Tian, Chao Huang, Chao Zhang, Chen Qian, Chenchen Ju, Chenchen Li, Chengfu Tang, Chilin Fu, Chunshao Ren, Chunwei Wu, Cong Zhang, Cunyin Peng, Dafeng Xu, Daixin Wang, Dalong Zhang, Dingnan Jin, Dingyuan Zhu, Dongke Hu, Fangzheng Zhao, Feifan Wu, Feng Zhu, Gangshan Wang, Haitao Zhang, Hailin Zhao, Hanxiao Zhang, Hanzi Wang, Hao Qian, Haoyi Yu, Heng Zhang, Hongliang Zhang, Hongzhi Luan, Huirong Dong, Huizhong Li, Jia Li, Jia Liu, Jialong Zhu, Jian Sha, Jianping Wei, Jiaolong Yang, Jieyue Ma, Jiewei Wu, Jinjing Huang, Jingyun Tian, Jingyuan Zhang, Jinquan Sun, Juanhui Tu, Jun Liu, Jun Xu, Jun Zhou, Junjie Ou, Junpeng Fang, Kaihong Zhang, Kaiqin Hu, Ke Shi, Kun Tang, Kunlong Chen, Lanyin Mei, Lei Liang, Lei Xu, Libo Zhang, Lin Ju, Lin Yuan, Ling Zhong, Lintao Ma, Lu Liu, Lu Yu, Lun Cai, Meiqi Zhu, Mengying Li, Min Chen, Minghao Xue, Minghong Cai, Mingming Yin, Peijie Jiang, Peilong Zhao, Pingping Liu, Qian Zhao, Qing Cui, Qingxiang Huang, Qingyuan Yang, Quankun Yu, Shaowei Wei, Shijie Lian, Shoujian Zheng, Shun Song, Shungen Zhang, Shuo Zhang, Siyuan Li, Song Liu, Ting Guo, Tong Zhao, Wanli Gu, Weichang Wu, Weiguang Han, Wenjing Fang, Wubin Wang, Xiang Shu, Xiao Shi, Xiaoshun Lan, Xiaolu Zhang, Xiaqing Sun, Xin Zhao, Xingyu Lu, Xiong Xu, Xudong Wang, Xudong Wang, Xuemin Yang, Yajie Yang, Yang Xiang, Yanzhe Li, Yi Zhang, Yilong Wang, Yingxue Li, Yongzhen Guo, Yuzhuo Fu, Yuanyuan Wang, Yue Yang, Yue Yu, Yufeng Deng, Yun Zhang, Yunfei Yu, Yuqi Zhang, Yuxiao He, Zengke Gui, Zhaoxin Huan, Zhaoyang Wang, Zhibo Zhu, Zhihao Wang, Zhiqiang Zhang, Zhoufei Wang, Zihang Zeng, Ziqi Liu, Zitao Xuan, Zuoli Tang  
**Category**: cs.AI  
**Published**: 2025-11-11  
**Score**: 9.0  
**Type**: replace-cross  
**ArXiv ID**: 2510.22115v2  

We introduce Ling 2.0, a series reasoning-oriented language foundation built upon the principle that every activation boosts reasoning capability. Designed to scale from tens of billions to one trillion parameters under a unified Mixture-of-Experts (MoE) paradigm, Ling 2.0 emphasizes high sparsity, ...

---

### 11. [SnapStream: Efficient Long Sequence Decoding on Dataflow Accelerators](https://arxiv.org/abs/2511.03092)

**Authors**: Jonathan Li, Nasim Farahini, Evgenii Iuliugin, Magnus Vesterlund, Christian H\"aggstr\"om, Guangtao Wang, Shubhangi Upasani, Ayush Sachdeva, Rui Li, Faline Fu, Chen Wu, Ayesha Siddiqua, John Long, Tuowen Zhao, Matheen Musaddiq, H\r{a}kan Zeffer, Yun Du, Mingran Wang, Qinghua Li, Bo Li, Urmish Thakker, Raghu Prabhakar  
**Category**: cs.DC  
**Published**: 2025-11-11  
**Score**: 9.0  
**Type**: replace-cross  
**ArXiv ID**: 2511.03092v3  

The proliferation of 100B+ parameter Large Language Models (LLMs) with 100k+ context length support have resulted in increasing demands for on-chip memory to support large KV caches. Techniques such as StreamingLLM and SnapKV demonstrate how to control KV cache size while maintaining model accuracy....

---

### 12. [Privacy-Preserving Federated Learning for Fair and Efficient Urban Traffic Optimization](https://arxiv.org/abs/2511.06363)

**Authors**: Rathin Chandra Shit, Sharmila Subudhi  
**Category**: cs.LG  
**Published**: 2025-11-11  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2511.06363v1  

The optimization of urban traffic is threatened by the complexity of achieving a balance between transport efficiency and the maintenance of privacy, as well as the equitable distribution of traffic based on socioeconomically diverse neighborhoods. Current centralized traffic management schemes inva...

---

### 13. [Breaking the Gradient Barrier: Unveiling Large Language Models for Strategic Classification](https://arxiv.org/abs/2511.06979)

**Authors**: Xinpeng Lv, Yunxin Mao, Haoxuan Li, Ke Liang, Jinxuan Yang, Wanrong Huang, Haoang Chi, Huan Chen, Long Lan, Yuanlong Chen, Wenjing Yang, Haotian Wang  
**Category**: cs.LG  
**Published**: 2025-11-11  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2511.06979v1  

Strategic classification~(SC) explores how individuals or entities modify their features strategically to achieve favorable classification outcomes. However, existing SC methods, which are largely based on linear models or shallow neural networks, face significant limitations in terms of scalability...

---

### 14. [Advanced Hybrid Transformer LSTM Technique with Attention and TS Mixer for Drilling Rate of Penetration Prediction](https://arxiv.org/abs/2508.05210)

**Authors**: Saddam Hussain Khan (Artificial Intelligence Lab, Department of Computer Systems Engineering, University of Engineering,Applied Sciences)  
**Category**: cs.AI  
**Published**: 2025-11-11  
**Score**: 8.5  
**Type**: replace-cross  
**ArXiv ID**: 2508.05210v3  

Rate of Penetration (ROP) prediction is critical for drilling optimization yet remains challenging due to the nonlinear, dynamic, and heterogeneous characteristics of drilling data. Conventional empirical, physics-based, and standard machine learning models rely on oversimplified assumptions or inte...

---

### 15. [Towards Resource-Efficient Multimodal Intelligence: Learned Routing among Specialized Expert Models](https://arxiv.org/abs/2511.06441)

**Authors**: Mayank Saini, Arit Kumar Bishwas  
**Category**: cs.CL  
**Published**: 2025-11-11  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2511.06441v1  

As AI moves beyond text, large language models (LLMs) increasingly power vision, audio, and document understanding; however, their high inference costs hinder real-time, scalable deployment. Conversely, smaller open-source models offer cost advantages but struggle with complex or multimodal queries....

---

### 16. [SR-KI: Scalable and Real-Time Knowledge Integration into LLMs via Supervised Attention](https://arxiv.org/abs/2511.06446)

**Authors**: Bohan Yu, Wei Huang, Kang Liu  
**Category**: cs.CL  
**Published**: 2025-11-11  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2511.06446v1  

This paper proposes SR-KI, a novel approach for integrating real-time and large-scale structured knowledge bases (KBs) into large language models (LLMs). SR-KI begins by encoding KBs into key-value pairs using a pretrained encoder, and injects them into LLMs' KV cache. Building on this representatio...

---

### 17. [DynaSpec: Context-aware Dynamic Speculative Sampling for Large-Vocabulary Language Models](https://arxiv.org/abs/2510.13847)

**Authors**: Jinbin Zhang, Nasib Ullah, Erik Schultheis, Rohit Babbar  
**Category**: cs.CL  
**Published**: 2025-11-11  
**Score**: 8.5  
**Type**: replace  
**ArXiv ID**: 2510.13847v2  

Speculative decoding has become a standard way to accelerate LLM inference: a small drafter proposes multiple tokens and a large target model verifies them once per speculation length. Recently, scaling of the LLM vocabulary has pushed the number of tokens to grow substantially. While verification o...

---

### 18. [Kunlun Anomaly Troubleshooter: Enabling Kernel-Level Anomaly Detection and Causal Reasoning for Large Model Distributed Inference](https://arxiv.org/abs/2511.05978)

**Authors**: Yuyang Liu, Jingjing Cai, Jiayi Ren, Peng Zhou, Danyang Zhang, Yin Du, Shijian Li  
**Category**: cs.DC  
**Published**: 2025-11-11  
**Score**: 8.5  
**Type**: cross  
**ArXiv ID**: 2511.05978v1  

Anomaly troubleshooting for large model distributed inference (LMDI) remains a critical challenge. Resolving anomalies such as inference performance degradation or latency jitter in distributed system demands significant manual efforts from domain experts, resulting in extremely time-consuming diagn...

---

### 19. [Beyond Redundancy: Diverse and Specialized Multi-Expert Sparse Autoencoder](https://arxiv.org/abs/2511.05745)

**Authors**: Zhen Xu, Zhen Tan, Song Wang, Kaidi Xu, Tianlong Chen  
**Category**: cs.LG  
**Published**: 2025-11-11  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2511.05745v1  

Sparse autoencoders (SAEs) have emerged as a powerful tool for interpreting large language models (LLMs) by decomposing token activations into combinations of human-understandable features. While SAEs provide crucial insights into LLM explanations, their practical adoption faces a fundamental challe...

---

### 20. [TNT: Improving Chunkwise Training for Test-Time Memorization](https://arxiv.org/abs/2511.07343)

**Authors**: Zeman Li, Ali Behrouz, Yuan Deng, Peilin Zhong, Praneeth Kacham, Mahdi Karami, Meisam Razaviyayn, Vahab Mirrokni  
**Category**: cs.LG  
**Published**: 2025-11-11  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2511.07343v1  

Recurrent neural networks (RNNs) with deep test-time memorization modules, such as Titans and TTT, represent a promising, linearly-scaling paradigm distinct from Transformers. While these expressive models do not yet match the peak performance of state-of-the-art Transformers, their potential has be...

---

### 21. [RedOne 2.0: Rethinking Domain-specific LLM Post-Training in Social Networking Services](https://arxiv.org/abs/2511.07070)

**Authors**: Fei Zhao, Chonggang Lu, Haofu Qian, Fangcheng Shi, Zijie Meng, Jianzhao Huang, Xu Tang, Zheyong Xie, Zheyu Ye, Zhe Xu, Yao Hu, Shaosheng Cao  
**Category**: cs.LG  
**Published**: 2025-11-11  
**Score**: 8.5  
**Type**: cross  
**ArXiv ID**: 2511.07070v1  

As a key medium for human interaction and information exchange, social networking services (SNS) pose unique challenges for large language models (LLMs): heterogeneous workloads, fast-shifting norms and slang, and multilingual, culturally diverse corpora that induce sharp distribution shift. Supervi...

---

### 22. [SONIQ: System-Optimized Noise-Injected Ultra-Low-Precision Quantization with Full-Precision Parity](https://arxiv.org/abs/2311.14114)

**Authors**: Cyrus Zhou, Pedro Savarese, Zack Hassman, Vaughn Richard, Michael DiBrino, Michael Maire, Yanjing Li  
**Category**: cs.LG  
**Published**: 2025-11-11  
**Score**: 8.5  
**Type**: replace-cross  
**ArXiv ID**: 2311.14114v4  

Ultra-low-precision inference can sharply reduce memory and latency but often degrades accuracy and relies on specialized hardware. We present SONIQ, a system-optimized, noise-injected quantization framework that learns per-channel mixed precision for both weights and activations while training unde...

---

### 23. [PuzzleMoE: Efficient Compression of Large Mixture-of-Experts Models via Sparse Expert Merging and Bit-packed inference](https://arxiv.org/abs/2511.04805)

**Authors**: Yushu Zhao, Zheng Wang, Minjia Zhang  
**Category**: cs.AI  
**Published**: 2025-11-11  
**Score**: 8.0  
**Type**: cross  
**ArXiv ID**: 2511.04805v1  

Mixture-of-Experts (MoE) models have shown strong potential in scaling language models efficiently by activating only a small subset of experts per input. However, their widespread deployment remains limited due to the high memory overhead associated with storing all expert parameters, particularly ...

---

### 24. [An End-to-End Deep Reinforcement Learning Approach for Solving the Traveling Salesman Problem with Drones](https://arxiv.org/abs/2511.05265)

**Authors**: Taihelong Zeng, Yun Lin, Yuhe Shi, Yan Li, Zhiqing Wei, Xuanru Ji  
**Category**: cs.AI  
**Published**: 2025-11-11  
**Score**: 8.0  
**Type**: cross  
**ArXiv ID**: 2511.05265v1  

The emergence of truck-drone collaborative systems in last-mile logistics has positioned the Traveling Salesman Problem with Drones (TSP-D) as a pivotal extension of classical routing optimization, where synchronized vehicle coordination promises substantial operational efficiency and reduced enviro...

---

### 25. [Introducing LongCat-Flash-Thinking: A Technical Report](https://arxiv.org/abs/2509.18883)

**Authors**: Meituan LongCat Team, Anchun Gui, Bei Li, Bingyang Tao, Bole Zhou, Borun Chen, Chao Zhang, Chao Zhang, Chengcheng Han, Chenhui Yang, Chi Zhang, Chong Peng, Chuyu Zhang, Cong Chen, Fengcun Li, Gang Xu, Guoyuan Lin, Hao Jiang, Hao Liang, Haomin Fu, Haoxiang Ma, Hong Liu, Hongyan Hao, Hongyin Tang, Hongyu Zang, Hongzhi Ni, Hui Su, Jiahao Liu, Jiahuan Li, Jialin Liu, Jianfei Zhang, Jianhao Xu, Jianing Wang, Jiaqi Sun, Jiaqi Zhang, Jiarong Shi, Jiawei Yang, Jingang Wang, Jinrui Ding, Jun Kuang, Jun Xu, Ke He, Kefeng Zhang, Keheng Wang, Keqing He, Li Wei, Liang Shi, Lin Qiu, Lingbin Kong, Lingchuan Liu, Linsen Guo, Longfei An, Mai Xia, Meng Zhou, Mengshen Zhu, Peng Pei, Pengcheng Jia, Qi Gu, Qi Guo, Qiong Huang, Quan Chen, Quanchi Weng, Rongxiang Weng, Ruichen Shao, Rumei Li, Shanglin Lei, Shuai Du, Shuaikang Liu, Shuang Zhou, Shuhao Hu, Siyu Xu, Songshan Gong, Tao Liang, Tianhao Hu, Wei He, Wei Shi, Wei Wang, Wei Wu, Wei Zhuo, Weifeng Tang, Wenjie Shi, Wenlong Zhu, Xi Su, Xiangcheng Liu, Xiangyu Xi, Xiangzhou Huang, Xiao Liu, Xiaochen Jiang, Xiaowei Shi, Xiaowen Shi, Xiaoyu Li, Xin Chen, Xinyue Zhao, Xuan Huang, Xuemiao Zhang, Xuezhi Cao, Xunliang Cai, Yajie Zhang, Yang Chen, Yang Liu, Yang Liu, Yang Zheng, Yaoming Wang, Yaqi Huo, Yerui Sun, Yifan Lu, Yiyang Li, Youshao Xiao, Yuanzhe Lei, Yuchen Xie, Yueqing Sun, Yufei Zhang, Yuhuai Wei, Yulei Qian, Yunke Zhao, Yuqing Ding, Yuwei Jiang, Zhaohua Yang, Zhengyu Chen, Zhijian Liu, Zhikang Xia, Zhongda Su, Ziran Li, Ziwen Wang, Ziyuan Zhuang, Zongyu Wang, Zunyuan Yang  
**Category**: cs.AI  
**Published**: 2025-11-11  
**Score**: 8.0  
**Type**: replace  
**ArXiv ID**: 2509.18883v2  

We present LongCat-Flash-Thinking, an efficient 560-billion-parameter open-source Mixture-of-Experts (MoE) reasoning model. Its advanced capabilities are cultivated through a meticulously crafted training process, beginning with long Chain-of-Thought (CoT) data cold-start and culminating in large-sc...

---

### 26. [Orion-MSP: Multi-Scale Sparse Attention for Tabular In-Context Learning](https://arxiv.org/abs/2511.02818)

**Authors**: Mohamed Bouadi, Pratinav Seth, Aditya Tanna, Vinay Kumar Sankarapu  
**Category**: cs.AI  
**Published**: 2025-11-11  
**Score**: 8.0  
**Type**: replace  
**ArXiv ID**: 2511.02818v3  

Tabular data remain the predominant format for real-world applications. Yet, developing effective neural models for tabular data remains challenging due to heterogeneous feature types and complex interactions occurring at multiple scales. Recent advances in tabular in-context learning (ICL), such as...

---

### 27. [C3PO: Optimized Large Language Model Cascades with Probabilistic Cost Constraints for Reasoning](https://arxiv.org/abs/2511.07396)

**Authors**: Antonios Valkanas, Soumyasundar Pal, Pavel Rumiantsev, Yingxue Zhang, Mark Coates  
**Category**: cs.LG  
**Published**: 2025-11-11  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2511.07396v1  

Large language models (LLMs) have achieved impressive results on complex reasoning tasks, but their high inference cost remains a major barrier to real-world deployment. A promising solution is to use cascaded inference, where small, cheap models handle easy queries, and only the hardest examples ar...

---

### 28. [LiveStar: Live Streaming Assistant for Real-World Online Video Understanding](https://arxiv.org/abs/2511.05299)

**Authors**: Zhenyu Yang, Kairui Zhang, Yuhang Hu, Bing Wang, Shengsheng Qian, Bin Wen, Fan Yang, Tingting Gao, Weiming Dong, Changsheng Xu  
**Category**: cs.AI  
**Published**: 2025-11-11  
**Score**: 7.5  
**Type**: cross  
**ArXiv ID**: 2511.05299v1  

Despite significant progress in Video Large Language Models (Video-LLMs) for offline video understanding, existing online Video-LLMs typically struggle to simultaneously process continuous frame-by-frame inputs and determine optimal response timing, often compromising real-time responsiveness and na...

---

### 29. [Confidence-Guided Stepwise Model Routing for Cost-Efficient Reasoning](https://arxiv.org/abs/2511.06190)

**Authors**: Sangmook Lee, Dohyung Kim, Hyukhun Koh, Nakyeong Yang, Kyomin Jung  
**Category**: cs.CL  
**Published**: 2025-11-11  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2511.06190v1  

Recent advances in Large Language Models (LLMs) - particularly model scaling and test-time techniques - have greatly enhanced the reasoning capabilities of language models at the expense of higher inference costs. To lower inference costs, prior works train router models or deferral mechanisms that ...

---

### 30. [Revisiting the Data Sampling in Multimodal Post-training from a Difficulty-Distinguish View](https://arxiv.org/abs/2511.06722)

**Authors**: Jianyu Qi, Ding Zou, Wenrui Yan, Rui Ma, Jiaxu Li, Zhijie Zheng, Zhiguo Yang, Rongchang Zhao  
**Category**: cs.CL  
**Published**: 2025-11-11  
**Score**: 7.5  
**Type**: cross  
**ArXiv ID**: 2511.06722v1  

Recent advances in Multimodal Large Language Models (MLLMs) have spurred significant progress in Chain-of-Thought (CoT) reasoning. Building on the success of Deepseek-R1, researchers extended multimodal reasoning to post-training paradigms based on reinforcement learning (RL), focusing predominantly...

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

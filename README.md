# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2025-12-02 05:32:57 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [SIMPLE: Disaggregating Sampling from GPU Inference into a Decision Plane for Faster Distributed LLM Serving](https://arxiv.org/abs/2512.00719)

**Authors**: Bohan Zhao, Zane Cao, Yongchao He  
**Category**: cs.DC  
**Published**: 2025-12-02  
**Score**: 14.0  
**Type**: new  
**ArXiv ID**: 2512.00719v1  

#### Abstract
As large language models (LLMs) scale out with tensor parallelism (TP) and pipeline parallelism (PP) and production stacks have aggressively optimized the data plane (attention/GEMM and KV cache), sampling, the decision plane that turns logits into tokens, becomes a new bottleneck. This creates a st...

#### AI Summary (by moonshot-v1-32k)
### 论文的主要贡献和创新点

本论文提出了SIMPLE，这是一个创新的架构，它将大型语言模型（LLM）的采样过程从GPU推理中分离出来，作为一个独立的决策平面服务。主要贡献和创新点包括：

1. **识别采样瓶颈**：论文首次识别并量化了在分布式LLM推理中，采样作为一个结构性的瓶颈问题，随着GPU性能的提升和词汇表的扩大，采样在迭代时间中所占比例不断增加。

2. **决策平面服务**：提出了一种新的决策平面服务，它与数据平面优化正交，遵循推测-然后校正的范式，包括序列并行、CPU卸载和基于Zipf分布的推测性热词采样。

3. **序列并行采样**：通过在批处理维度上分片工作，避免了词汇表轴上的集体操作，实现了与张量并行（TP）友好的并行性。

4. **CPU基础算法**：实现了基于列的惩罚和截断优先过滤的单次线性时间内核，以减少内存流量并实现与GPU计算的重叠。

5. **推测性热词采样（SHVS）**：利用Zipf分布，通过在热门子词汇表上采样，并使用拒绝校正来保持输出分布的准确性，从而提高了吞吐量。

### 核心实验方法和设置

实验在L40、H100和B200节点上进行，涵盖了不同的GPU型号和配置。选择了在分布式配置下服务的模型，并在每种平台上选择了最大化吞吐量的TP/PP度数。实验中使用了16个采样器和4个线程，并在不同的负载下评估了性能。

### 主要实验结果和性能指标

实验结果显示：

1. **吞吐量提升**：SIMPLE在不同设备上相比于GPU上的基线平均提升了50%的端到端吞吐量，最高可达96%。

2. **延迟降低**：在P95延迟方面，SIMPLE在不同设备上平均降低了55%，最高可达65%。

3. **资源利用率**：SIMPLE提高了GPU利用率，平均提升了21%，并且在CPU上的利用也有所增加，但仍然远未达到饱和。

4. **主机内存使用**：SIMPLE增加了主机内存的使用，但增幅较小，平均增加了0.8%。

### 关键结论和发现

论文的关键结论和发现包括：

1. **采样瓶颈**：采样过程已经成为分布式LLM推理中的一个结构性瓶颈，它不随张量并行扩展，也不在流水线并行中平衡，并且随着GPU的加速和词汇表的增长，其在迭代时间中所占比例不断增加。

2. **性能提升**：SIMPLE通过将采样过程从GPU数据平面中分离出来，并作为一个独立的决策平面服务运行，显著提高了端到端的吞吐量和降低了延迟。

3. **资源利用**：SIMPLE在提高GPU利用率的同时，对CPU和主机内存的额外需求很小，表明该架构在现代硬件上是可行的。

4. **分布准确性**：SIMPLE的SHVS方法在提高性能的同时，保持了输出分布的准确性，没有牺牲采样质量。

总的来说，SIMPLE通过重新架构采样过程，使其与现代硬件扩展相一致，移除了顽固的串行尾部，解锁了推理性能的下一个增量，而无需用户代码更改。

---

### 2. [RL-Struct: A Lightweight Reinforcement Learning Framework for Reliable Structured Output in LLMs](https://arxiv.org/abs/2512.00319)

**Authors**: Ruike Hu, Shulei Wu  
**Category**: cs.AI  
**Published**: 2025-12-02  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2512.00319v1  

#### Abstract
Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language generation and reasoning. However, their integration into automated software ecosystems is often hindered by the "Structure Gap" - the inherent tension between the probabilistic nature of token generation and ...

#### AI Summary (by moonshot-v1-32k)
### 1. 论文的主要贡献和创新点

这篇论文提出了一个名为RL-Struct的轻量级强化学习框架，旨在解决大型语言模型（LLMs）在生成可靠结构化输出（如JSON、XML）时面临的“结构鸿沟”问题。主要贡献和创新点包括：

- **多维奖励函数**：提出了一个新颖的多维奖励函数，将结构化输出任务分解为结构完整性、格式正确性、内容准确性和有效性等多个层次的约束。
- **GRPO优化**：利用梯度正则化策略优化（Gradient Regularized Policy Optimization, GRPO），使模型能够在没有单独的批评网络的情况下内化这些约束，减少了峰值VRAM使用量40%。
- **自适应课程学习**：通过实验分析揭示了模型在训练过程中自然形成的自适应课程学习现象，即模型先掌握语法熟练度，再细化语义准确性。
- **高效RL框架**：展示了一个高效的RL框架，其中GRPO结合低秩适应（LoRA）提供了一种稳定且计算效率高的方法，以使LLMs与结构约束对齐。

### 2. 核心实验方法和设置

- **数据集**：使用了“AkashPS11/recipes_data_food.com”数据集，过滤出高质量的示例，任务是生成具有特定字段的JSON对象。
- **基线模型**：与多种最新的模型进行比较，包括闭源专有模型、开源通用模型、高效的小型语言模型以及受限解码和对齐模型。
- **训练**：使用LoRA进行训练，学习率为$5 \times 10^{-6}$，采用余弦衰减计划。
- **评估指标**：除了标准的结构化指标外，还采用了LLM作为独立评判的协议，使用GPT-4-Turbo评估语义正确性。

### 3. 主要实验结果和性能指标

- **结构准确性**：RL-Struct方法在结构准确性上达到了89.7%，显著优于基线模型。
- **JSON有效性**：在JSON有效性上达到了92.1%，同样显著优于其他模型。
- **格式一致性**：在格式一致性上达到了85.3%。
- **模式符合性**：在模式符合性上达到了89.7%。
- **内容准确性**：在内容准确性上达到了84.5%，展现了在结构化输出任务中的优越性能。

### 4. 关键结论和发现

- **RL在结构化输出中的有效性**：强化学习信号作为一种非可微正则化器，能够惩罚即使是微小的语法偏差，使模型学习到更健壮的内部表示。
- **内部化与约束**：与在推理时通过屏蔽无效标记来保证语法正确的受限解码方法相比，RL调整的模型“学习”结构，允许更快的推理和更好的适应新模式。
- **自适应课程学习**：模型在训练过程中自然地优先优化“更容易”的结构奖励，然后再处理“更难”的语义目标，无需手动设计课程计划。
- **泛化能力**：RL-Struct方法在多个任务中保持了高结构准确性，表明模型已经获得了结构化输出原则的稳健表示，有助于有效迁移到未见过的模式。

这篇论文展示了通过强化学习来优化LLMs生成结构化输出的有效性，并提出了一种新的框架来弥合概率性AI和确定性软件工程之间的差距。

---

### 3. [SpeContext: Enabling Efficient Long-context Reasoning with Speculative Context Sparsity in LLMs](https://arxiv.org/abs/2512.00722)

**Authors**: Jiaming Xu, Jiayi Pan, Hanzhen Wang, Yongkang Zhou, Jiancai Ye, Yu Wang, Guohao Dai  
**Category**: cs.AI  
**Published**: 2025-12-02  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2512.00722v1  

#### Abstract
In this paper, we point out that the objective of the retrieval algorithms is to align with the LLM, which is similar to the objective of knowledge distillation in LLMs. We analyze the similarity in information focus between the distilled language model(DLM) and the original LLM from the perspective...

#### AI Summary (by moonshot-v1-32k)
### 1. 论文的主要贡献和创新点

本篇论文的主要贡献在于提出了一个名为SpeContext的算法和系统联合设计，旨在提高大型语言模型（LLMs）在长文本上下文中的推理效率。创新点包括：

- **信息检索算法的新颖范式**：通过分析蒸馏语言模型（DLM）和原始LLM之间的信息焦点相似性，提出利用DLM作为检索算法的新范式，以高效检索重要信息焦点。
- **轻量级检索头设计**：基于DLM的头级注意力权重，设计了轻量级检索头，通过剪枝冗余操作，实现了超过90%的参数减少。
- **异步预取数据流**：设计了通过弹性加载策略的异步预取数据流，有效重叠了KV缓存检索与LLM计算。
- **自适应内存管理系统**：构建了理论内存模型，并实现了一个自适应内存管理系统，通过最大化GPU内存利用率来加速推理。

### 2. 核心实验方法和设置

实验在两种资源受限环境中进行评估：云端的高端GPU多请求环境和边缘的低端GPU有限内存环境。实验比较了SpeContext与几个最新的KV缓存优化工作以及典型的LLM框架（如Huggingface）和LLM推理引擎（如FlashInfer）的性能。

- **硬件平台**：选择了配备NVIDIA A100-80GB GPU的工作站和配备NVIDIA RTX 4060 Laptop GPU的联想Legion Y7000PC。
- **基线**：选择了Huggingface和FlashInfer作为全注意力的基线，以及Quest、ClusterKV和ShadowKV作为稀疏注意力的基线。
- **模型和基准**：选择了Llama3.1-8B、DeepSeek-R1-Distill-Llama-8B和Qwen3-8B等LLMs进行评估，并使用了LongBench和LongWriter等基准测试。

### 3. 主要实验结果和性能指标

- **云端环境**：SpeContext相较于全注意力的Huggingface框架，实现了高达24.89倍的吞吐量提升，与最先进的FlashInfer相比，也有2.20倍的提升。
- **边缘环境**：在边缘环境中，SpeContext相较于全注意力的Huggingface框架，实现了高达10.06倍的速度提升，与最先进的ShadowKV相比，也有1.17倍的提升。
- **准确性**：在长文本输入和推理场景中，SpeContext在不同的KV预算下保持了与全注意力相当的准确性，仅在KV预算较小时略有下降。

### 4. 关键结论和发现

- SpeContext通过算法、系统和编译层面的优化，有效地提高了LLMs在长文本上下文推理中的效率，同时保持了可忽略的准确性损失。
- 通过实验验证，SpeContext在资源受限的环境中，如云端和边缘设备，均能显著提升推理速度和吞吐量，推动了准确性和吞吐量的帕累托前沿。
- 该研究的方法论和视角可以扩展到更多考虑信息的机器学习架构和系统设计的进一步研究中。

---

### 4. [Efficient and Programmable Exploration of Synthesizable Chemical Space](https://arxiv.org/abs/2512.00384)

**Authors**: Shitong Luo, Connor W. Coley  
**Category**: cs.LG  
**Published**: 2025-12-02  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2512.00384v1  

#### Abstract
The constrained nature of synthesizable chemical space poses a significant challenge for sampling molecules that are both synthetically accessible and possess desired properties. In this work, we present PrexSyn, an efficient and programmable model for molecular discovery within synthesizable chemic...

#### AI Summary (by moonshot-v1-32k)
### 1. 论文的主要贡献和创新点

本论文的主要贡献在于提出了PrexSyn，这是一个高效且可编程的模型，用于在可合成的化学空间中发现分子。PrexSyn基于仅解码器的变换器（decoder-only transformer），通过大规模数据流训练，能够近乎完美地重建可合成的化学空间，并学习属性与可合成分子之间的关联。其创新点包括：

- **高效的数据生成引擎**：通过实时、高通量的C++基础数据生成引擎，实现了亿级别数据流的训练，提高了训练的规模和效率。
- **属性条件生成**：PrexSyn能够根据单一属性提示或复合属性查询生成满足条件的可合成分子，允许用户通过逻辑运算符“编程”生成目标。
- **查询空间优化**：PrexSyn通过迭代查询细化，高效优化分子对抗黑盒预言机函数，比合成不可知的基线具有更高的采样效率。

### 2. 核心实验方法和设置

- **模型架构**：PrexSyn采用仅解码器的变换器架构，输入分子属性提示，自回归生成合成路径的后缀表示。
- **数据生成**：使用C++开发的多线程数据管道，实时生成合成路径并计算分子属性，提高了训练数据的生成效率。
- **训练设置**：模型使用Enamine的现货构建块集和Gao等人策划的反应模板集进行训练，包含12个变换层，模型维度为1024，前馈维度为2048，注意力头数为16。
- **性能评估**：通过化学空间投影任务和GuacaMol基准测试套件评估PrexSyn的性能，包括重建率、相似性和AUC-Top10得分等指标。

### 3. 主要实验结果和性能指标

- **化学空间投影**：PrexSyn在Enamine测试集上达到了94.06%的重建率和0.9859的Tanimoto相似性得分，显著优于先前的方法。
- **合成分子采样效率**：在GuacaMol基准测试中，PrexSyn在8个任务中的6个上取得了最高的平均得分，超过了所有合成不可知和合成基础的基线。
- **复合属性查询**：PrexSyn能够根据复合属性查询生成满足特定条件的分子，如Lipinski's Rule of Five的模拟药物发现场景。
- **对接预言机优化**：在sEH和Mpro2任务中，PrexSyn生成的分子在对接得分上优于基线抑制剂，并且具有更好的药物相似性（QED得分）。

### 4. 关键结论和发现

- PrexSyn通过其高效的数据引擎、近乎完美的化学空间覆盖和查询空间优化，推动了可合成分子设计的前沿。
- PrexSyn的可编程性体现在支持复合属性查询和基于查询能力的黑盒预言机函数优化上，使其成为分子设计和优化的强大工具。
- 实验结果表明，PrexSyn在准确性、速度和采样效率方面均达到了新的最佳水平，展示了其在实际药物发现和化学空间探索中的潜力。

---

### 5. [Financial Text Classification Based On rLoRA Finetuning On Qwen3-8B model](https://arxiv.org/abs/2512.00630)

**Authors**: Zhiming Lian  
**Category**: cs.LG  
**Published**: 2025-12-02  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2512.00630v1  

#### Abstract
Financial text classification has increasingly become an important aspect in quantitative trading systems and related tasks, such as financial sentiment analysis and the classification of financial news. In this paper, we assess the performance of the large language model Qwen3-8B on both tasks. Qwe...

#### AI Summary (by moonshot-v1-32k)
### 1. 论文的主要贡献和创新点

这篇论文的主要贡献在于评估和展示了Qwen3-8B大型语言模型在金融文本分类任务上的性能，特别是在金融情感分析和金融新闻分类两个方面。创新点包括：

- **模型优化**：Qwen3-8B模型针对金融应用进行了特别优化，具有高效的微调和高性能推理能力。
- **微调方法**：应用了Noisy Embedding Instruction Finetuning（NEFTune）和Rank-stabilized Low-Rank Adaptation（rLoRA）技术，提高了模型的鲁棒性，并减少了GPU内存的使用，使得大型模型的微调更加高效。
- **FlashAttention技术**：进一步加快了训练速度，并降低了内存使用。

### 2. 核心实验方法和设置

- **数据集**：使用了两个数据集，一个用于金融情感分类，包含近3000个样本；另一个是Twitter上的金融新闻数据集，用于分类金融新闻文章。
- **模型架构**：Qwen3-8B模型具有82亿参数和36个Transformer层，支持长达32K的上下文窗口。
- **微调技术**：采用了rLoRA技术进行参数高效的微调，以及FlashAttention技术来加速注意力机制的计算。
- **实验设置**：设置了包括批大小、梯度累积步数、学习率、最大令牌长度等在内的一系列超参数。

### 3. 主要实验结果和性能指标

- **性能对比**：Qwen3-8B在金融情感分类和金融主题分类任务上均超过了传统的Transformer模型（如BERT、RoBERTa）以及其他大型模型（如LLaMA1-7B、LLaMA2-7B、Baichuan2-7B）。
- **准确率**：Qwen3-8B在情感分类上达到了0.8415的准确率，在主题分类上达到了0.9315的准确率。
- **训练效率**：Qwen3-8B在三个训练周期内就实现了稳定性能，与传统非LLM方法相比，后者通常需要超过十个周期。

### 4. 关键结论和发现

- **模型性能**：Qwen3-8B在金融文本分类任务上展现出了优越的性能，不仅准确率高，而且训练效率高。
- **微调策略**：结合基于指令的微调、噪声正则化的嵌入和参数高效的微调策略，Qwen3-8B在降低计算成本的同时提高了泛化能力。
- **实际应用**：Qwen3-8B不仅在金融NLP基准测试中表现良好，而且在实际应用中，如量化交易系统，也具有实用性和可扩展性，特别是在准确性、效率和适应性方面。

---

### 6. [Domain-Decomposed Graph Neural Network Surrogate Modeling for Ice Sheets](https://arxiv.org/abs/2512.01888)

**Authors**: Adrienne M. Propp, Mauro Perego, Eric C. Cyr, Anthony Gruber, Amanda A. Howard, Alexander Heinlein, Panos Stinis, Daniel M. Tartakovsky  
**Category**: cs.LG  
**Published**: 2025-12-02  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2512.01888v1  

#### Abstract
Accurate yet efficient surrogate models are essential for large-scale simulations of partial differential equations (PDEs), particularly for uncertainty quantification (UQ) tasks that demand hundreds or thousands of evaluations. We develop a physics-inspired graph neural network (GNN) surrogate that...

---

### 7. [FlexiWalker: Extensible GPU Framework for Efficient Dynamic Random Walks with Runtime Adaptation](https://arxiv.org/abs/2512.00705)

**Authors**: Seongyeon Park, Jaeyong Song, Changmin Shin, Sukjin Kim, Junguk Hong, Jinho Lee  
**Category**: cs.DC  
**Published**: 2025-12-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2512.00705v1  

#### Abstract
Dynamic random walks are fundamental to various graph analysis applications, offering advantages by adapting to evolving graph properties. Their runtime-dependent transition probabilities break down the pre-computation strategy that underpins most existing CPU and GPU static random walk optimization...

---

### 8. [Efficient Training of Diffusion Mixture-of-Experts Models: A Practical Recipe](https://arxiv.org/abs/2512.01252)

**Authors**: Yahui Liu, Yang Yue, Jingyuan Zhang, Chenxi Sun, Yang Zhou, Wencong Zeng, Ruiming Tang, Guorui Zhou  
**Category**: cs.LG  
**Published**: 2025-12-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2512.01252v1  

#### Abstract
Recent efforts on Diffusion Mixture-of-Experts (MoE) models have primarily focused on developing more sophisticated routing mechanisms. However, we observe that the underlying architectural configuration space remains markedly under-explored. Inspired by the MoE design paradigms established in large...

---

### 9. [KV Pareto: Systems-Level Optimization of KV Cache and Model Compression for Long Context Inference](https://arxiv.org/abs/2512.01953)

**Authors**: Sai Gokhale, Devleena Das, Rajeev Patwari, Ashish Sirasao, Elliott Delaye  
**Category**: cs.LG  
**Published**: 2025-12-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2512.01953v1  

#### Abstract
Long-context Large Language Models (LLMs) face significant memory bottlenecks during inference due to the linear growth of key-value (KV) cache with sequence length. While individual optimization techniques like KV cache quantization, chunked prefill, and model weight quantization have shown promise...

---

### 10. [Heimdall++: Optimizing GPU Utilization and Pipeline Parallelism for Efficient Single-Pulse Detection](https://arxiv.org/abs/2512.00398)

**Authors**: Bingzheng Xia, Zujie Ren, Kuang Ma, Xiaoqian Li, Wenda Li, Shuibing He  
**Category**: cs.DC  
**Published**: 2025-12-02  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2512.00398v1  

#### Abstract
With the increasing time and frequency resolution of modern radio telescopes and the exponential growth in observational data volumes, real-time single-pulse detection has become a critical requirement for time-domain radio astronomy. Heimdall, as a representative GPU-accelerated single-pulse search...

---

### 11. [EDIT: Early Diffusion Inference Termination for dLLMs Based on Dynamics of Training Gradients](https://arxiv.org/abs/2512.00670)

**Authors**: He-Yen Hsieh, Hong Wang, H. T. Kung  
**Category**: cs.AI  
**Published**: 2025-12-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2512.00670v1  

#### Abstract
Diffusion-based large language models (dLLMs) refine token generations through iterative denoising, but answers often stabilize before all steps complete. We propose EDIT (Early Diffusion Inference Termination), an inference-time criterion that adaptively stops denoising once sufficient reasoning st...

---

### 12. [Multi-Path Collaborative Reasoning via Reinforcement Learning](https://arxiv.org/abs/2512.01485)

**Authors**: Jindi Lv, Yuhao Zhou, Zheng Zhu, Xiaofeng Wang, Guan Huang, Jiancheng Lv  
**Category**: cs.AI  
**Published**: 2025-12-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2512.01485v1  

#### Abstract
Chain-of-Thought (CoT) reasoning has significantly advanced the problem-solving capabilities of Large Language Models (LLMs), yet conventional CoT often exhibits internal determinism during decoding, limiting exploration of plausible alternatives. Recent methods attempt to address this by generating...

---

### 13. [Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling](https://arxiv.org/abs/2512.02010)

**Authors**: Jack Cook, Junxian Guo, Guangxuan Xiao, Yujun Lin, Song Han  
**Category**: cs.CL  
**Published**: 2025-12-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2512.02010v1  

#### Abstract
As large language models have grown larger, low-precision numerical formats such as NVFP4 have become increasingly popular due to the speed and memory benefits they provide. However, to accelerate computation with NVFP4, all matrix multiplication operands--weights and activations in the forward pass...

---

### 14. [A Parallel and Distributed Rust Library for Core Decomposition on Large Graphs](https://arxiv.org/abs/2512.00233)

**Authors**: Davide Rucci, Sebastian Parfeniuc, Matteo Mordacchini, Emanuele Carlini, Alfredo Cuzzocrea, Patrizio Dazzi  
**Category**: cs.DC  
**Published**: 2025-12-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2512.00233v1  

#### Abstract
In this paper, we investigate the parallelization of $k$-core decomposition, a method used in graph analysis to identify cohesive substructures and assess node centrality. Although efficient sequential algorithms exist for this task, the scale of modern networks requires faster, multicore-ready appr...

---

### 15. [Clinical-R1: Empowering Large Language Models for Faithful and Comprehensive Reasoning with Clinical Objective Relative Policy Optimization](https://arxiv.org/abs/2512.00601)

**Authors**: Boyang Gu, Hongjian Zhou, Bradley Max Segal, Jinge Wu, Zeyu Cao, Hantao Zhong, Lei Clifton, Fenglin Liu, David A. Clifton  
**Category**: cs.AI  
**Published**: 2025-12-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2512.00601v1  

#### Abstract
Recent advances in large language models (LLMs) have shown strong reasoning capabilities through large-scale pretraining and post-training reinforcement learning, demonstrated by DeepSeek-R1. However, current post-training methods, such as Grouped Relative Policy Optimization (GRPO), mainly reward c...

---

### 16. [ARCADIA: Scalable Causal Discovery for Corporate Bankruptcy Analysis Using Agentic AI](https://arxiv.org/abs/2512.00839)

**Authors**: Fabrizio Maturo, Donato Riccio, Andrea Mazzitelli, Giuseppe Bifulco, Francesco Paolone, Iulia Brezeanu  
**Category**: cs.AI  
**Published**: 2025-12-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2512.00839v1  

#### Abstract
This paper introduces ARCADIA, an agentic AI framework for causal discovery that integrates large-language-model reasoning with statistical diagnostics to construct valid, temporally coherent causal structures. Unlike traditional algorithms, ARCADIA iteratively refines candidate DAGs through constra...

---

### 17. [Probabilistic Neuro-Symbolic Reasoning for Sparse Historical Data: A Framework Integrating Bayesian Inference, Causal Models, and Game-Theoretic Allocation](https://arxiv.org/abs/2512.01723)

**Authors**: Saba Kublashvili  
**Category**: cs.AI  
**Published**: 2025-12-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2512.01723v1  

#### Abstract
Modeling historical events poses fundamental challenges for machine learning: extreme data scarcity (N << 100), heterogeneous and noisy measurements, missing counterfactuals, and the requirement for human interpretable explanations. We present HistoricalML, a probabilistic neuro-symbolic framework t...

---

### 18. [Steady and Energy-Efficient Multi-Hop Clustering for Flying Ad-Hoc Networks (FANETs)](https://arxiv.org/abs/2512.00623)

**Authors**: Basilis Mamalis, Marios Perlitis  
**Category**: cs.DC  
**Published**: 2025-12-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2512.00623v1  

#### Abstract
Flying Ad-hoc Networks (FANETs), formed by Unmanned Aerial Vehicles (UAVs), represent an emerging and promising communication paradigm. These networks face unique challenges due to UAVs high mobility, limited energy resources, and dynamic topology. In this work, we propose a novel multi-hop clusteri...

---

### 19. [Scalable and Interpretable Scientific Discovery via Sparse Variational Gaussian Process Kolmogorov-Arnold Networks (SVGP KAN)](https://arxiv.org/abs/2512.00260)

**Authors**: Y. Sungtaek Ju  
**Category**: cs.LG  
**Published**: 2025-12-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2512.00260v1  

#### Abstract
Kolmogorov-Arnold Networks (KANs) offer a promising alternative to Multi-Layer Perceptron (MLP) by placing learnable univariate functions on network edges, enhancing interpretability. However, standard KANs lack probabilistic outputs, limiting their utility in applications requiring uncertainty quan...

---

### 20. [Upcycled and Merged MoE Reward Model for Mitigating Reward Hacking](https://arxiv.org/abs/2512.00724)

**Authors**: Lingling Fu  
**Category**: cs.LG  
**Published**: 2025-12-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2512.00724v1  

#### Abstract
Reward models play a critical role in Reinforcement Learning from Human Feedback (RLHF) by assessing the consistency between generated outputs and human preferences. However, conventional reward models are prone to reward hacking or over-optimization, where the policy exploits shortcut patterns to o...

---

### 21. [When Human Preferences Flip: An Instance-Dependent Robust Loss for RLHF](https://arxiv.org/abs/2512.00709)

**Authors**: Yifan Xu, Xichen Ye, Yifan Chen, Qiaosheng Zhang  
**Category**: cs.AI  
**Published**: 2025-12-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.00709v1  

#### Abstract
Quality of datasets plays an important role in large language model (LLM) alignment. In collecting human feedback, however, preference flipping is ubiquitous and causes corruption in data annotation; the issue necessitates the alignment algorithms with improved robustness against potential flipped p...

---

### 22. [Conveying Imagistic Thinking in Traditional Chinese Medicine Translation: A Prompt Engineering and LLM-Based Evaluation Framework](https://arxiv.org/abs/2512.01198)

**Authors**: Jiatong Han  
**Category**: cs.CL  
**Published**: 2025-12-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.01198v1  

#### Abstract
Traditional Chinese Medicine (TCM) theory is built on imagistic thinking, in which medical principles and diagnostic and therapeutic logic are structured through metaphor and metonymy. However, existing English translations largely rely on literal rendering, making it difficult for target-language r...

---

### 23. [Elastic Mixture of Rank-Wise Experts for Knowledge Reuse in Federated Fine-Tuning](https://arxiv.org/abs/2512.00902)

**Authors**: Yebo Wu, Jingguang Li, Zhijiang Guo, Li Li  
**Category**: cs.DC  
**Published**: 2025-12-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.00902v1  

#### Abstract
Federated fine-tuning offers a promising solution for adapting Large Language Models (LLMs) to downstream tasks while safeguarding data privacy. However, its high computational and communication demands hinder its deployment on resource-constrained devices. In this paper, we propose SmartFed, a reso...

---

### 24. [Hybrid Context-Fusion Attention (CFA) U-Net and Clustering for Robust Seismic Horizon Interpretation](https://arxiv.org/abs/2512.00191)

**Authors**: Jose Luis Lima de Jesus Silva, Joao Pedro Gomes, Paulo Roberto de Melo Barros Junior, Vitor Hugo Serravalle Reis Rodrigues, Alexsandro Guerra Cerqueira  
**Category**: cs.LG  
**Published**: 2025-12-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.00191v1  

#### Abstract
Interpreting seismic horizons is a critical task for characterizing subsurface structures in hydrocarbon exploration. Recent advances in deep learning, particularly U-Net-based architectures, have significantly improved automated horizon tracking. However, challenges remain in accurately segmenting ...

---

### 25. [Projection-Free CNN Pruning via Frank-Wolfe with Momentum: Sparser Models with Less Pretraining](https://arxiv.org/abs/2512.01147)

**Authors**: Hamza ElMokhtar Shili, Natasha Patnaik, Isabelle Ruble, Kathryn Jarjoura, Daniel Suarez Aguirre  
**Category**: cs.LG  
**Published**: 2025-12-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.01147v1  

#### Abstract
We investigate algorithmic variants of the Frank-Wolfe (FW) optimization method for pruning convolutional neural networks. This is motivated by the "Lottery Ticket Hypothesis", which suggests the existence of smaller sub-networks within larger pre-trained networks that perform comparatively well (if...

---

### 26. [Sum Rate Maximization in STAR-RIS-UAV-Assisted Networks: A CA-DDPG Approach for Joint Optimization](https://arxiv.org/abs/2512.01202)

**Authors**: Yujie Huang, Haibin Wan, Xiangcheng Li, Tuanfa Qin, Yun Li, Jun Li, Wen Chen  
**Category**: cs.LG  
**Published**: 2025-12-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.01202v1  

#### Abstract
With the rapid advances in programmable materials, reconfigurable intelligent surfaces (RIS) have become a pivotal technology for future wireless communications. The simultaneous transmitting and reflecting reconfigurable intelligent surfaces (STAR-RIS) can both transmit and reflect signals, enablin...

---

### 27. [Efficient Hyperparameter Search for Non-Stationary Model Training](https://arxiv.org/abs/2512.01258)

**Authors**: Berivan Isik, Matthew Fahrbach, Dima Kuzmin, Nicolas Mayoraz, Emil Praun, Steffen Rendle, Raghavendra Vasudeva  
**Category**: cs.LG  
**Published**: 2025-12-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.01258v1  

#### Abstract
Online learning is the cornerstone of applications like recommendation and advertising systems, where models continuously adapt to shifting data distributions. Model training for such systems is remarkably expensive, a cost that multiplies during hyperparameter search. We introduce a two-stage parad...

---

### 28. [Forget Less, Retain More: A Lightweight Regularizer for Rehearsal-Based Continual Learning](https://arxiv.org/abs/2512.01818)

**Authors**: Lama Alssum, Hasan Abed Al Kader Hammoud, Motasem Alfarra, Juan C Leon Alcazar, Bernard Ghanem  
**Category**: cs.LG  
**Published**: 2025-12-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.01818v1  

#### Abstract
Deep neural networks suffer from catastrophic forgetting, where performance on previous tasks degrades after training on a new task. This issue arises due to the model's tendency to overwrite previously acquired knowledge with new information. We present a novel approach to address this challenge, f...

---

### 29. [SemAgent: Semantic-Driven Agentic AI Empowered Trajectory Prediction in Vehicular Networks](https://arxiv.org/abs/2512.00834)

**Authors**: Lin Zhu, Kezhi Wang, Luping Xiang, Kun Yang  
**Category**: cs.AI  
**Published**: 2025-12-02  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2512.00834v1  

#### Abstract
Efficient information exchange and reliable contextual reasoning are essential for vehicle-to-everything (V2X) networks. Conventional communication schemes often incur significant transmission overhead and latency, while existing trajectory prediction models generally lack environmental perception a...

---

### 30. [CLIP-RL: Aligning Language and Policy Representations for Task Transfer in Reinforcement Learning](https://arxiv.org/abs/2512.01616)

**Authors**: Chainesh Gautam, Raghuram Bharadwaj Diddigi  
**Category**: cs.AI  
**Published**: 2025-12-02  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2512.01616v1  

#### Abstract
Recently, there has been an increasing need to develop agents capable of solving multiple tasks within the same environment, especially when these tasks are naturally associated with language. In this work, we propose a novel approach that leverages combinations of pre-trained (language, policy) pai...

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

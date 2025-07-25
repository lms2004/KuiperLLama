项目名称：模型推理框架开发
时间：xxxx年yy月 - 至今
项目描述：设计并构建了一个高效、灵活且可扩展的深度学习模型推理框架，旨在实现提供快速、准确的模型推理服务。通过优化模型加载和推理流程，大幅提升了处理速度和资源利用效率，能有效支撑多种业务需求。
职责与贡献：
1. 架构搭建：成功实现了对支持 Llama 2/3.2和Qwen 等主流深度学习模型的推理框架搭建，支持包括 YOLOv5、ResNet 在内的多种主流模型，支持CPU，GPU双后端推理，支持KV-Cache等常见优化手段（这里可以自己扩展Continous batching, PageAttention，投机采样等项目中没有的技术，但是自己去学一下并且知道在项目中实现就行）。
2. 算子调优：
  1. 对 Llama 2/3.2和Qwen 模型进行推理性能优化，针对 CUDA 算子进行了深度定制和优化，包括实现 RMSNorm、MatMul、KV - Cache 以及 MultiHead Attention ，FlashAttention等核心算子。
  2. 利用 Night - Compute 和 Night - System 等工具对算子性能进行细致分析和调优，确保最佳计算性能。经调优，在同等算力资源下，模型推理吞吐量提升了[X]% 。应用 Flash Attention、KV Cache 等技术，显著提高推理效率。在 xxxx硬件上，将采用 Int8 量化后的 LLAMA3.2-8B 推理速度提升至 yy Token/s，显存占用优化至 zz GB。（这里可以换个说法，可以说，模型的首字时延降低至[X]毫秒，平均时延降至[X]毫秒，相比优化前分别降低了[X]%和[X]% 。）
3. 量化技术应用：引入 int8 分组量化技术，对模型权重进行量化处理。通过精心调整分组策略和量化参数，在保证模型精度损失控制在极小范围内（精度下降仅[X]% ）的同时，大幅降低了模型内存占用，压缩比达到[X]，在配套开发动态反量化核情况下，显著提升了推理速度，进一步优化了模型在硬件设备上的运行效率。
项目成果：
1. 完成 Llama 2/3.2和Qwen 模型中多个复杂 CUDA 算子的开发与性能优化，涵盖 RMSNorm、MatMul、KV - Cache 和 MHA 等，分析并优化了内存管理技术，显著提高模型推理速度和资源利用率。优化后，内存占用峰值降低了[X]% ，性能提供了[Y]%。
2. 深入了解并熟练运用多种大型模型推理框架，如 llama.cpp、VLLM 及 TensorRT 等，对 llama.cpp 框架进行深入研究，熟练掌握其内部实现和优化技巧。 这里你可以列举下VLLM的优化技术，比如Multi-step scheduling，Chunked prefill，Speculative decoding，PagedAttention，Continuous batching，自己按名词学习一下即可。
3. 成功应用 int8 分组量化技术，为模型推理框架在资源受限环境下的高效运行提供了有力支持，为项目拓展了更广泛的应用场景。 这里可以扩展一下AWQ量化技术。



具体推理速度
1. llama2 :stories110M.bin tokenizer.model
电源
[Generation] Tokens: 989, Time: 3.2111 sec, TPS: 307.99 tokens/sec
[Overall] Total time: 3.2407 sec

steps/s:308.565568

电池
[Generation] Tokens: 989, Time: 4.4382 sec, TPS: 222.84 tokens/sec
[Overall] Total time: 4.4795 sec

steps/s:223.233205

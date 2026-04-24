---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:573
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
metrics:
- accuracy
- accuracy_threshold
- f1
- f1_threshold
- precision
- recall
- average_precision
model-index:
- name: CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2
  results:
  - task:
      type: cross-encoder-binary-classification
      name: Cross Encoder Binary Classification
    dataset:
      name: failure driven val
      type: failure-driven-val
    metrics:
    - type: accuracy
      value: 0.6984126984126984
      name: Accuracy
    - type: accuracy_threshold
      value: 0.3714524805545807
      name: Accuracy Threshold
    - type: f1
      value: 0.7710843373493976
      name: F1
    - type: f1_threshold
      value: 0.3714524805545807
      name: F1 Threshold
    - type: precision
      value: 0.6274509803921569
      name: Precision
    - type: recall
      value: 1.0
      name: Recall
    - type: average_precision
      value: 0.5918242296918768
      name: Average Precision
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision c5ee24cb16019beea0893ab7796b1df96625c6b8 -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

### Full Model Architecture

```
CrossEncoder(
  (0): Transformer({'transformer_task': 'sequence-classification', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'logits'}}, 'module_output_name': 'scores', 'architecture': 'BertForSequenceClassification'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of inputs
pairs = [
    ['What tools are commonly used to build RAG systems?', 'informed response . Here ’ s a quick illustration : The external data is converted into embedding vectors with a separate embeddings model , and the vectors are kept in a database . Embeddings models are typically small , so updating the embedding vectors on a regular basis is faster , cheaper , and easier than fine - tuning a model . At the same time , the fact that fine - tuning is not required gives you the freedom to swap your LLM for a more powerful one when it becomes available , or switch to a smaller distilled version , should you need faster inference . Let ’ s illustrate building a RAG using an open - source LLM , embeddings model , and LangChain . First , install the required dependencies : ! pip install - q'],
    ['What tools are commonly used to build RAG systems?', '- party data while simultaneously granting models access to it — access that can be revoked at any time . However , enterprises must be vigilant to maintain the security of the external databases themselves . RAG uses vector databases , which use embeddings to convert data points to numerical representations . If these databases are breached , attackers can reverse the vector embedding process and access the original data , especially if the vector database is unencrypted . Get curated insights on the most important — and intriguing — AI news . Subscribe to our weekly Think newsletter . See the IBM Privacy Statement . RAG systems essentially enable users to query databases with conversational language . The data - powered question - answering abilities of RAG systems have been applied across a range of use cases , including'],
    ['How can caching improve RAG performance?', 'nDCG @ K ) metric , commonly used in IR [ 7 ] , is less useful in the context of RAG because the rank order of retrieved documents is less important . It has been shown that as long as the relevant tokens are present , and the overall context is of reasonable length , LLMs can accurately process the information regardless of token position [ 8 ] . To address these shortcomings , we propose a new evaluation designed to capture the essential details of retrieval performance in the AI application context , consisting of a generative dataset , and a new performance measure . We describe the pipeline that generates the dataset , allowing others to generate domain specific evaluations for their own data and use cases . Because the dataset is generated , in general it'],
    ['What is an ablation study for RAG evaluation?', 'the metrics though and RAG agents , let ’ s recap what RAG is . A RAG pipeline is an architecture where an LLM ’ s output is informed by external data that is retrieved at runtime based on an input . Rather than relying solely on the model ’ s trained knowledge , a RAG system first : - Searches a knowledge source — like a document database , vector store , or API , then - Feeds the retrieved content into the prompt for the LLM to generate a response . Here ’ s a diagram showing how RAG works : A RAG Pipeline Architecture You ’ ll notice that the quality of the final generation is highly dependent on the retriever doing its job well . A RAG pipeline can only produce a helpful , factually correct'],
    ['How do you evaluate performance under noise?', 'applied broadly , see Evaluating Measures in Information Retrieval . For an analysis using these metrics to measure the performance of Pinecone Assistant , see Benchmarking AI Assistants . Understanding binary relevance metrics While different frameworks combine metrics and sometimes create custom metrics , grounding in common information retrieval formulas will help you evaluate which frameworks best suit your use cases . The following is not an exhaustive list of available metrics but a good starting point . The retrieval metrics we ’ re examining first fit under the binary relevance umbrella , where a result is either relevant or irrelevant . There are two categories of metrics that observe binary relevance : Order - unaware and Order - aware . Order - unaware metrics examine if results are relevant and correct , regardless of which order they ’ re'],
]
scores = model.predict(pairs)
print(scores)
# [ 4.6508 -1.4385 -4.3472 -4.3386 -2.0632]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'What tools are commonly used to build RAG systems?',
    [
        'informed response . Here ’ s a quick illustration : The external data is converted into embedding vectors with a separate embeddings model , and the vectors are kept in a database . Embeddings models are typically small , so updating the embedding vectors on a regular basis is faster , cheaper , and easier than fine - tuning a model . At the same time , the fact that fine - tuning is not required gives you the freedom to swap your LLM for a more powerful one when it becomes available , or switch to a smaller distilled version , should you need faster inference . Let ’ s illustrate building a RAG using an open - source LLM , embeddings model , and LangChain . First , install the required dependencies : ! pip install - q',
        '- party data while simultaneously granting models access to it — access that can be revoked at any time . However , enterprises must be vigilant to maintain the security of the external databases themselves . RAG uses vector databases , which use embeddings to convert data points to numerical representations . If these databases are breached , attackers can reverse the vector embedding process and access the original data , especially if the vector database is unencrypted . Get curated insights on the most important — and intriguing — AI news . Subscribe to our weekly Think newsletter . See the IBM Privacy Statement . RAG systems essentially enable users to query databases with conversational language . The data - powered question - answering abilities of RAG systems have been applied across a range of use cases , including',
        'nDCG @ K ) metric , commonly used in IR [ 7 ] , is less useful in the context of RAG because the rank order of retrieved documents is less important . It has been shown that as long as the relevant tokens are present , and the overall context is of reasonable length , LLMs can accurately process the information regardless of token position [ 8 ] . To address these shortcomings , we propose a new evaluation designed to capture the essential details of retrieval performance in the AI application context , consisting of a generative dataset , and a new performance measure . We describe the pipeline that generates the dataset , allowing others to generate domain specific evaluations for their own data and use cases . Because the dataset is generated , in general it',
        'the metrics though and RAG agents , let ’ s recap what RAG is . A RAG pipeline is an architecture where an LLM ’ s output is informed by external data that is retrieved at runtime based on an input . Rather than relying solely on the model ’ s trained knowledge , a RAG system first : - Searches a knowledge source — like a document database , vector store , or API , then - Feeds the retrieved content into the prompt for the LLM to generate a response . Here ’ s a diagram showing how RAG works : A RAG Pipeline Architecture You ’ ll notice that the quality of the final generation is highly dependent on the retriever doing its job well . A RAG pipeline can only produce a helpful , factually correct',
        'applied broadly , see Evaluating Measures in Information Retrieval . For an analysis using these metrics to measure the performance of Pinecone Assistant , see Benchmarking AI Assistants . Understanding binary relevance metrics While different frameworks combine metrics and sometimes create custom metrics , grounding in common information retrieval formulas will help you evaluate which frameworks best suit your use cases . The following is not an exhaustive list of available metrics but a good starting point . The retrieval metrics we ’ re examining first fit under the binary relevance umbrella , where a result is either relevant or irrelevant . There are two categories of metrics that observe binary relevance : Order - unaware and Order - aware . Order - unaware metrics examine if results are relevant and correct , regardless of which order they ’ re',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Cross Encoder Binary Classification

* Dataset: `failure-driven-val`
* Evaluated with [<code>CEBinaryClassificationEvaluator</code>](https://sbert.net/docs/package_reference/cross_encoder/evaluation.html#sentence_transformers.cross_encoder.evaluation.CEBinaryClassificationEvaluator)

| Metric                | Value      |
|:----------------------|:-----------|
| accuracy              | 0.6984     |
| accuracy_threshold    | 0.3715     |
| f1                    | 0.7711     |
| f1_threshold          | 0.3715     |
| precision             | 0.6275     |
| recall                | 1.0        |
| **average_precision** | **0.5918** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 573 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 573 samples:
  |         | sentence_0                                                                        | sentence_1                                                                           | label                                                         |
  |:--------|:----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                            | string                                                                               | float                                                         |
  | details | <ul><li>min: 7 tokens</li><li>mean: 10.88 tokens</li><li>max: 13 tokens</li></ul> | <ul><li>min: 142 tokens</li><li>mean: 162.6 tokens</li><li>max: 302 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.5</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                      | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | label            |
  |:----------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>What tools are commonly used to build RAG systems?</code> | <code>informed response . Here ’ s a quick illustration : The external data is converted into embedding vectors with a separate embeddings model , and the vectors are kept in a database . Embeddings models are typically small , so updating the embedding vectors on a regular basis is faster , cheaper , and easier than fine - tuning a model . At the same time , the fact that fine - tuning is not required gives you the freedom to swap your LLM for a more powerful one when it becomes available , or switch to a smaller distilled version , should you need faster inference . Let ’ s illustrate building a RAG using an open - source LLM , embeddings model , and LangChain . First , install the required dependencies : ! pip install - q</code>                                                                                                            | <code>1.0</code> |
  | <code>What tools are commonly used to build RAG systems?</code> | <code>- party data while simultaneously granting models access to it — access that can be revoked at any time . However , enterprises must be vigilant to maintain the security of the external databases themselves . RAG uses vector databases , which use embeddings to convert data points to numerical representations . If these databases are breached , attackers can reverse the vector embedding process and access the original data , especially if the vector database is unencrypted . Get curated insights on the most important — and intriguing — AI news . Subscribe to our weekly Think newsletter . See the IBM Privacy Statement . RAG systems essentially enable users to query databases with conversational language . The data - powered question - answering abilities of RAG systems have been applied across a range of use cases , including</code> | <code>0.0</code> |
  | <code>How can caching improve RAG performance?</code>           | <code>nDCG @ K ) metric , commonly used in IR [ 7 ] , is less useful in the context of RAG because the rank order of retrieved documents is less important . It has been shown that as long as the relevant tokens are present , and the overall context is of reasonable length , LLMs can accurately process the information regardless of token position [ 8 ] . To address these shortcomings , we propose a new evaluation designed to capture the essential details of retrieval performance in the AI application context , consisting of a generative dataset , and a new performance measure . We describe the pipeline that generates the dataset , allowing others to generate domain specific evaluations for their own data and use cases . Because the dataset is generated , in general it</code>                                                                 | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 3
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: trackio
- `per_device_eval_batch_size`: 16
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch | Step | failure-driven-val_average_precision |
|:-----:|:----:|:------------------------------------:|
| 1.0   | 36   | 0.4256                               |
| 2.0   | 72   | 0.4703                               |
| 3.0   | 108  | 0.5918                               |


### Training Time
- **Training**: 7.1 seconds

### Framework Versions
- Python: 3.12.3
- Sentence Transformers: 5.4.1
- Transformers: 5.5.4
- PyTorch: 2.7.1+cu126
- Accelerate: 1.13.0
- Datasets: 4.8.4
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->
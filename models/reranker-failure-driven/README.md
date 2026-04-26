---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:447
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L12-v2
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
- name: CrossEncoder based on cross-encoder/ms-marco-MiniLM-L12-v2
  results:
  - task:
      type: cross-encoder-binary-classification
      name: Cross Encoder Binary Classification
    dataset:
      name: failure driven val
      type: failure-driven-val
    metrics:
    - type: accuracy
      value: 1.0
      name: Accuracy
    - type: accuracy_threshold
      value: 1.2924163341522217
      name: Accuracy Threshold
    - type: f1
      value: 1.0
      name: F1
    - type: f1_threshold
      value: 1.2924163341522217
      name: F1 Threshold
    - type: precision
      value: 1.0
      name: Precision
    - type: recall
      value: 1.0
      name: Recall
    - type: average_precision
      value: 1.0
      name: Average Precision
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L12-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2) <!-- at revision 7b0235231ca2674cb8ca8f022859a6eba2b1c968 -->
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
    ['What are good retrieval metrics for RAG?', 'relevant documents or data . RAG evaluation quantifies the accuracy of your retrieval phrase by calculating metrics on the top results your system returns , enabling you to programmatically monitor your pipeline ’ s precision , recall ability , and faithfulness to facts . First , we ’ ll examine some of the most commonly used metrics and how they are derived . Then , we ’ ll survey frameworks and tooling that employ these metrics to quantify the performance of your RAG deployment . Finally , we ’ ll help you choose the best framework and tooling for your use case to ensure your RAG deployments consistently achieve your performance goals . Note : this chapter focuses on RAG pipelines . For an in - depth treatment of Information Retrieval metrics applied broadly , see Evaluating Measures in Information'],
    ['Can structured and unstructured data be combined?', 'application for the user to read . RAG data pipeline flow The following workflow describes a high - level flow for a data pipeline that supplies grounding data for a RAG application . - Documents or other media are either pushed or pulled into a data pipeline . - The data pipeline processes each media file individually by completing the following steps : - Chunking : Breaks down the media file into semantically relevant parts that ideally have a single idea or concept . - Enrich chunks : Adds metadata fields that the pipeline creates based on the content in the chunks . The data pipeline categorizes the metadata into discrete fields , such as title , summary , and keywords . - Embed chunks : Uses an embedding model to vectorize the chunk and any other metadata fields that'],
    ['How can you reduce token usage in RAG?', 'Open - Source AI Cookbook documentation Advanced RAG on Hugging Face documentation using LangChain Advanced RAG on Hugging Face documentation using LangChain Authored by : Aymeric Roucher This notebook demonstrates how you can build an advanced RAG ( Retrieval Augmented Generation ) for answering a user ’ s question about a specific knowledge base ( here , the HuggingFace documentation ) , using LangChain . For an introduction to RAG , you can check this other cookbook ! RAG systems are complex , with many moving parts : here is a RAG diagram , where we noted in blue all possibilities for system enhancement : 💡 As you can see , there are many steps to tune in this architecture : tuning the system properly will yield significant performance gains . In this notebook , we will take a look'],
    ['What are the main components of a RAG system?', ". For agentic RAG use cases , which we ' ll cover more in a later section , you might also find it useful to include a task completion metric to evaluate your AI agent RAG pipeline as well . Before we get too into the metrics though and RAG agents , let ’ s recap what RAG is . A RAG pipeline is an architecture where an LLM ’ s output is informed by external data that is retrieved at runtime based on an input . Rather than relying solely on the model ’ s trained knowledge , a RAG system first : - Searches a knowledge source — like a document database , vector store , or API , then - Feeds the retrieved content into the prompt for the LLM to generate a response . Here ’"],
    ['What are good retrieval metrics for RAG?', "’ s contribution to the final response quality To do this , RAG evaluation involves 5 key industry - standard metrics : - Answer Relevancy : How relevant the generated response is to the given input . - Faithfulness : Whether the generated response contains hallucinations to the retrieval context . - Contextual Relevancy : How relevant the retrieval context is to the input . - Contextual Recall : Whether the retrieval context contains all the information required to produce the ideal output ( for a given input ) . - Contextual Precision : Whether the retrieval context is ranked in the correct order ( higher relevancy goes first ) for a given input . For agentic RAG use cases , which we ' ll cover more in a later section , you might also find it useful to include"],
]
scores = model.predict(pairs)
print(scores)
# [-6.4667  7.7422 -8.9841 -6.1198 -6.1198]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'What are good retrieval metrics for RAG?',
    [
        'relevant documents or data . RAG evaluation quantifies the accuracy of your retrieval phrase by calculating metrics on the top results your system returns , enabling you to programmatically monitor your pipeline ’ s precision , recall ability , and faithfulness to facts . First , we ’ ll examine some of the most commonly used metrics and how they are derived . Then , we ’ ll survey frameworks and tooling that employ these metrics to quantify the performance of your RAG deployment . Finally , we ’ ll help you choose the best framework and tooling for your use case to ensure your RAG deployments consistently achieve your performance goals . Note : this chapter focuses on RAG pipelines . For an in - depth treatment of Information Retrieval metrics applied broadly , see Evaluating Measures in Information',
        'application for the user to read . RAG data pipeline flow The following workflow describes a high - level flow for a data pipeline that supplies grounding data for a RAG application . - Documents or other media are either pushed or pulled into a data pipeline . - The data pipeline processes each media file individually by completing the following steps : - Chunking : Breaks down the media file into semantically relevant parts that ideally have a single idea or concept . - Enrich chunks : Adds metadata fields that the pipeline creates based on the content in the chunks . The data pipeline categorizes the metadata into discrete fields , such as title , summary , and keywords . - Embed chunks : Uses an embedding model to vectorize the chunk and any other metadata fields that',
        'Open - Source AI Cookbook documentation Advanced RAG on Hugging Face documentation using LangChain Advanced RAG on Hugging Face documentation using LangChain Authored by : Aymeric Roucher This notebook demonstrates how you can build an advanced RAG ( Retrieval Augmented Generation ) for answering a user ’ s question about a specific knowledge base ( here , the HuggingFace documentation ) , using LangChain . For an introduction to RAG , you can check this other cookbook ! RAG systems are complex , with many moving parts : here is a RAG diagram , where we noted in blue all possibilities for system enhancement : 💡 As you can see , there are many steps to tune in this architecture : tuning the system properly will yield significant performance gains . In this notebook , we will take a look',
        ". For agentic RAG use cases , which we ' ll cover more in a later section , you might also find it useful to include a task completion metric to evaluate your AI agent RAG pipeline as well . Before we get too into the metrics though and RAG agents , let ’ s recap what RAG is . A RAG pipeline is an architecture where an LLM ’ s output is informed by external data that is retrieved at runtime based on an input . Rather than relying solely on the model ’ s trained knowledge , a RAG system first : - Searches a knowledge source — like a document database , vector store , or API , then - Feeds the retrieved content into the prompt for the LLM to generate a response . Here ’",
        "’ s contribution to the final response quality To do this , RAG evaluation involves 5 key industry - standard metrics : - Answer Relevancy : How relevant the generated response is to the given input . - Faithfulness : Whether the generated response contains hallucinations to the retrieval context . - Contextual Relevancy : How relevant the retrieval context is to the input . - Contextual Recall : Whether the retrieval context contains all the information required to produce the ideal output ( for a given input ) . - Contextual Precision : Whether the retrieval context is ranked in the correct order ( higher relevancy goes first ) for a given input . For agentic RAG use cases , which we ' ll cover more in a later section , you might also find it useful to include",
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

| Metric                | Value   |
|:----------------------|:--------|
| accuracy              | 1.0     |
| accuracy_threshold    | 1.2924  |
| f1                    | 1.0     |
| f1_threshold          | 1.2924  |
| precision             | 1.0     |
| recall                | 1.0     |
| **average_precision** | **1.0** |

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

* Size: 447 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 447 samples:
  |         | sentence_0                                                                        | sentence_1                                                                            | label                                                         |
  |:--------|:----------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                            | string                                                                                | float                                                         |
  | details | <ul><li>min: 9 tokens</li><li>mean: 10.96 tokens</li><li>max: 12 tokens</li></ul> | <ul><li>min: 143 tokens</li><li>mean: 158.21 tokens</li><li>max: 237 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.5</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                     | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | label            |
  |:---------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>What are good retrieval metrics for RAG?</code>          | <code>relevant documents or data . RAG evaluation quantifies the accuracy of your retrieval phrase by calculating metrics on the top results your system returns , enabling you to programmatically monitor your pipeline ’ s precision , recall ability , and faithfulness to facts . First , we ’ ll examine some of the most commonly used metrics and how they are derived . Then , we ’ ll survey frameworks and tooling that employ these metrics to quantify the performance of your RAG deployment . Finally , we ’ ll help you choose the best framework and tooling for your use case to ensure your RAG deployments consistently achieve your performance goals . Note : this chapter focuses on RAG pipelines . For an in - depth treatment of Information Retrieval metrics applied broadly , see Evaluating Measures in Information</code> | <code>0.0</code> |
  | <code>Can structured and unstructured data be combined?</code> | <code>application for the user to read . RAG data pipeline flow The following workflow describes a high - level flow for a data pipeline that supplies grounding data for a RAG application . - Documents or other media are either pushed or pulled into a data pipeline . - The data pipeline processes each media file individually by completing the following steps : - Chunking : Breaks down the media file into semantically relevant parts that ideally have a single idea or concept . - Enrich chunks : Adds metadata fields that the pipeline creates based on the content in the chunks . The data pipeline categorizes the metadata into discrete fields , such as title , summary , and keywords . - Embed chunks : Uses an embedding model to vectorize the chunk and any other metadata fields that</code>                              | <code>1.0</code> |
  | <code>How can you reduce token usage in RAG?</code>            | <code>Open - Source AI Cookbook documentation Advanced RAG on Hugging Face documentation using LangChain Advanced RAG on Hugging Face documentation using LangChain Authored by : Aymeric Roucher This notebook demonstrates how you can build an advanced RAG ( Retrieval Augmented Generation ) for answering a user ’ s question about a specific knowledge base ( here , the HuggingFace documentation ) , using LangChain . For an introduction to RAG , you can check this other cookbook ! RAG systems are complex , with many moving parts : here is a RAG diagram , where we noted in blue all possibilities for system enhancement : 💡 As you can see , there are many steps to tune in this architecture : tuning the system properly will yield significant performance gains . In this notebook , we will take a look</code>                | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `num_train_epochs`: 4

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 8
- `num_train_epochs`: 4
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
- `per_device_eval_batch_size`: 8
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
| 1.0   | 56   | 0.7824                               |
| 2.0   | 112  | 1.0                                  |
| 3.0   | 168  | 1.0                                  |
| 4.0   | 224  | 1.0                                  |


### Training Time
- **Training**: 12.4 seconds

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
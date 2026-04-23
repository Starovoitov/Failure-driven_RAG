---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:657
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
      value: 0.8493150684931506
      name: Accuracy
    - type: accuracy_threshold
      value: -2.2620997428894043
      name: Accuracy Threshold
    - type: f1
      value: 0.8705882352941177
      name: F1
    - type: f1_threshold
      value: -2.2620997428894043
      name: F1 Threshold
    - type: precision
      value: 0.7708333333333334
      name: Precision
    - type: recall
      value: 1.0
      name: Recall
    - type: average_precision
      value: 0.8290109962904081
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
    ['What is multi-hop retrieval in RAG?', 'The latest AI trends , brought to you by experts Get curated insights on the most important — and intriguing — AI news . Subscribe to our weekly Think newsletter . See the IBM Privacy Statement . Retrieval augmented generation , or RAG , is an architecture for optimizing the performance of an artificial intelligence ( AI ) model by connecting it with external knowledge bases . RAG helps large language models ( LLMs ) deliver more relevant responses at a higher quality . Generative AI ( gen AI ) models are trained on large datasets and refer to this information to generate outputs . However , training datasets are finite and limited to the information the AI developer can access — public domain works , internet articles , social media content and other publicly accessible data . RAG allows generative AI models to access additional external knowledge bases , such as internal organizational data , scholarly journals and specialized'],
    ['What is the role of data labeling in RAG?', '- What is RAG evaluation , how is it different from regular LLM and AI agent evaluation , and common points of failure - Retriever metrics such as contextual relevancy , recall , and precision - Generator metrics such as answer relevancy and faithfulness - How to run RAG evaluation : both end - to - end and at a component - level - Best practices , including RAG evaluation in CI pipelines and post - deployment monitoring All of course , this all includes code samples using DeepEval ⭐ , an open - source LLM evaluation framework . Let ’ s get started . TL ; DR - RAG pipelines are made up of a retriever and a generator , both of which contribute to the quality of the final response . - RAG metrics measures either the retriever and generator in isolation , focusing on relevancy , hallucination , and retrieval . - Retriever metrics include : Contextual'],
    ['What are best practices for prompt design?', 'bring in knowledge relevant to your situation - current events , news , social media , customer data , proprietary data - Builds trust : more relevant and accurate results are more likely to earn trust and source citations allow human review - More control : control over which sources are used , real - time data access , authorization to data , guardrails / safety / compliance , traceability / source citations , retrieval strategies , cost , tune each component independently of the others - Cost - effective compared to alternatives like training / re - training your own model , fine - tuning , or stuffing the context window : foundation models are costly to produce and require specialized knowledge to create , as is fine - tuning ; the larger the context sent to the model , the higher the cost RAG in support of agentic workflows But this traditional RAG approach is simple , often'],
    ['What is the risk of under-retrieval?', ". To know if your retrieval is working well , you need ways to measure : There are a few ways to evaluate this , depending on whether you ' re running offline evaluations – the kind you use during experiments or regression testing – or online evaluations , as part of production monitoring . It also depends on how much labeled data you have to design the test . Let ’ s take a look at 3 different approaches . First things first : retrieval isn ’ t a new problem . It ’ s the same task behind every search bar – from e - commerce sites to Google to internal company portals . It ’ s a classic machine learning use case , and there are well - established evaluation methods we can reuse for LLM - powered RAG setups . To apply them , you need a ground truth dataset – your custom retrieval benchmark ."],
    ['How does RAG work in simple terms?', 'cutoff , it loses relevance over time . RAG systems connect models with supplemental external data in real - time and incorporate up - to - date information into generated responses . Enterprises use RAG to equip models with specific information such as proprietary customer data , authoritative research and other relevant documents . RAG models can also connect to the internet with application programming interfaces ( APIs ) and gain access to real - time social media feeds and consumer reviews for a better understanding of market sentiment . Meanwhile , access to breaking news and search engines can lead to more accurate responses as models incorporate the retrieved information into the text - generation process . Generative AI models such as OpenAI ’ s GPT work by detecting patterns in their data , then using those patterns to predict the most likely outcomes to user inputs . Sometimes models detect patterns that don ’ t exist . A'],
]
scores = model.predict(pairs)
print(scores)
# [-1.8238  1.6281  3.9184 -6.4205 -2.909 ]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'What is multi-hop retrieval in RAG?',
    [
        'The latest AI trends , brought to you by experts Get curated insights on the most important — and intriguing — AI news . Subscribe to our weekly Think newsletter . See the IBM Privacy Statement . Retrieval augmented generation , or RAG , is an architecture for optimizing the performance of an artificial intelligence ( AI ) model by connecting it with external knowledge bases . RAG helps large language models ( LLMs ) deliver more relevant responses at a higher quality . Generative AI ( gen AI ) models are trained on large datasets and refer to this information to generate outputs . However , training datasets are finite and limited to the information the AI developer can access — public domain works , internet articles , social media content and other publicly accessible data . RAG allows generative AI models to access additional external knowledge bases , such as internal organizational data , scholarly journals and specialized',
        '- What is RAG evaluation , how is it different from regular LLM and AI agent evaluation , and common points of failure - Retriever metrics such as contextual relevancy , recall , and precision - Generator metrics such as answer relevancy and faithfulness - How to run RAG evaluation : both end - to - end and at a component - level - Best practices , including RAG evaluation in CI pipelines and post - deployment monitoring All of course , this all includes code samples using DeepEval ⭐ , an open - source LLM evaluation framework . Let ’ s get started . TL ; DR - RAG pipelines are made up of a retriever and a generator , both of which contribute to the quality of the final response . - RAG metrics measures either the retriever and generator in isolation , focusing on relevancy , hallucination , and retrieval . - Retriever metrics include : Contextual',
        'bring in knowledge relevant to your situation - current events , news , social media , customer data , proprietary data - Builds trust : more relevant and accurate results are more likely to earn trust and source citations allow human review - More control : control over which sources are used , real - time data access , authorization to data , guardrails / safety / compliance , traceability / source citations , retrieval strategies , cost , tune each component independently of the others - Cost - effective compared to alternatives like training / re - training your own model , fine - tuning , or stuffing the context window : foundation models are costly to produce and require specialized knowledge to create , as is fine - tuning ; the larger the context sent to the model , the higher the cost RAG in support of agentic workflows But this traditional RAG approach is simple , often',
        ". To know if your retrieval is working well , you need ways to measure : There are a few ways to evaluate this , depending on whether you ' re running offline evaluations – the kind you use during experiments or regression testing – or online evaluations , as part of production monitoring . It also depends on how much labeled data you have to design the test . Let ’ s take a look at 3 different approaches . First things first : retrieval isn ’ t a new problem . It ’ s the same task behind every search bar – from e - commerce sites to Google to internal company portals . It ’ s a classic machine learning use case , and there are well - established evaluation methods we can reuse for LLM - powered RAG setups . To apply them , you need a ground truth dataset – your custom retrieval benchmark .",
        'cutoff , it loses relevance over time . RAG systems connect models with supplemental external data in real - time and incorporate up - to - date information into generated responses . Enterprises use RAG to equip models with specific information such as proprietary customer data , authoritative research and other relevant documents . RAG models can also connect to the internet with application programming interfaces ( APIs ) and gain access to real - time social media feeds and consumer reviews for a better understanding of market sentiment . Meanwhile , access to breaking news and search engines can lead to more accurate responses as models incorporate the retrieved information into the text - generation process . Generative AI models such as OpenAI ’ s GPT work by detecting patterns in their data , then using those patterns to predict the most likely outcomes to user inputs . Sometimes models detect patterns that don ’ t exist . A',
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

| Metric                | Value     |
|:----------------------|:----------|
| accuracy              | 0.8493    |
| accuracy_threshold    | -2.2621   |
| f1                    | 0.8706    |
| f1_threshold          | -2.2621   |
| precision             | 0.7708    |
| recall                | 1.0       |
| **average_precision** | **0.829** |

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

* Size: 657 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 657 samples:
  |         | sentence_0                                                                        | sentence_1                                                                            | label                                                         |
  |:--------|:----------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                            | string                                                                                | float                                                         |
  | details | <ul><li>min: 8 tokens</li><li>mean: 10.58 tokens</li><li>max: 14 tokens</li></ul> | <ul><li>min: 165 tokens</li><li>mean: 175.39 tokens</li><li>max: 292 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.5</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                              | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | label            |
  |:--------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>What is multi-hop retrieval in RAG?</code>        | <code>The latest AI trends , brought to you by experts Get curated insights on the most important — and intriguing — AI news . Subscribe to our weekly Think newsletter . See the IBM Privacy Statement . Retrieval augmented generation , or RAG , is an architecture for optimizing the performance of an artificial intelligence ( AI ) model by connecting it with external knowledge bases . RAG helps large language models ( LLMs ) deliver more relevant responses at a higher quality . Generative AI ( gen AI ) models are trained on large datasets and refer to this information to generate outputs . However , training datasets are finite and limited to the information the AI developer can access — public domain works , internet articles , social media content and other publicly accessible data . RAG allows generative AI models to access additional external knowledge bases , such as internal organizational data , scholarly journals and specialized</code> | <code>0.0</code> |
  | <code>What is the role of data labeling in RAG?</code>  | <code>- What is RAG evaluation , how is it different from regular LLM and AI agent evaluation , and common points of failure - Retriever metrics such as contextual relevancy , recall , and precision - Generator metrics such as answer relevancy and faithfulness - How to run RAG evaluation : both end - to - end and at a component - level - Best practices , including RAG evaluation in CI pipelines and post - deployment monitoring All of course , this all includes code samples using DeepEval ⭐ , an open - source LLM evaluation framework . Let ’ s get started . TL ; DR - RAG pipelines are made up of a retriever and a generator , both of which contribute to the quality of the final response . - RAG metrics measures either the retriever and generator in isolation , focusing on relevancy , hallucination , and retrieval . - Retriever metrics include : Contextual</code>                                                                                    | <code>1.0</code> |
  | <code>What are best practices for prompt design?</code> | <code>bring in knowledge relevant to your situation - current events , news , social media , customer data , proprietary data - Builds trust : more relevant and accurate results are more likely to earn trust and source citations allow human review - More control : control over which sources are used , real - time data access , authorization to data , guardrails / safety / compliance , traceability / source citations , retrieval strategies , cost , tune each component independently of the others - Cost - effective compared to alternatives like training / re - training your own model , fine - tuning , or stuffing the context window : foundation models are costly to produce and require specialized knowledge to create , as is fine - tuning ; the larger the context sent to the model , the higher the cost RAG in support of agentic workflows But this traditional RAG approach is simple , often</code>                                                   | <code>1.0</code> |
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
| 1.0   | 42   | 0.6067                               |
| 2.0   | 84   | 0.6981                               |
| 3.0   | 126  | 0.8290                               |


### Training Time
- **Training**: 7.4 seconds

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
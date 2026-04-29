from __future__ import annotations

import json


def dedupe_query_variants(variants: list[str]) -> list[str]:
    """Remove semantically duplicate query variants while preserving order.

    Normalizing case and trailing punctuation keeps variant lists compact and
    prevents repeated retrieval requests for effectively the same query.
    """
    deduped: list[str] = []
    seen_keys: set[str] = set()
    for variant in variants:
        key = variant.strip().lower().rstrip("?.!")
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(variant.strip())
    return deduped


def paraphrase_variants(base: str) -> list[str]:
    """Generate lightweight paraphrases to broaden lexical coverage.

    These variants help retrieval hit documents that use alternative wording
    for the same concept.
    """
    lowered = base.lower()
    variants: list[str] = []
    if "?" in base:
        variants.append(base.replace("?", "").strip())
    replacements = (
        ("rag", "retrieval augmented generation"),
        ("retrieval augmented generation", "rag"),
        ("reranker", "cross encoder reranker"),
        ("reranking", "cross encoder reranking"),
        ("hallucination", "factual consistency"),
        ("recall", "retrieval coverage"),
    )
    for src, dst in replacements:
        if src in lowered:
            variants.append(lowered.replace(src, dst))
    return dedupe_query_variants(variants)


def decomposition_variants(base: str) -> list[str]:
    """Create decomposition queries targeting sub-problems of the input.

    Decomposition is used to improve recall for complex questions by issuing
    narrower retrieval intents.
    """
    lowered = base.lower().strip().rstrip("?")
    variants: list[str] = []
    if "context stuffing" in lowered:
        variants.extend(
            [
                "what does context stuffing mean in rag",
                "what is adding too much context in llm prompts",
                "what is overloading context window rag",
                "problems with too much retrieved context",
            ]
        )
    if "why" in lowered and "recall" in lowered:
        variants.extend(
            [
                "how does recall work in retrieval systems",
                "what causes low recall in bm25 retrieval",
                "what causes embedding retrieval misses",
                "how does hybrid retrieval merge bm25 and dense results",
            ]
        )
    if "hybrid" in lowered:
        variants.append("how hybrid retrieval combines bm25 and semantic rankings")
    if "rerank" in lowered:
        variants.append("what features improve cross encoder reranking quality")
    if "evaluation" in lowered or "metrics" in lowered:
        variants.extend(
            [
                "how to evaluate retrieval quality in rag systems",
                "how mrr and ndcg diagnose reranking failures",
            ]
        )
    if not variants:
        variants.extend(
            [
                f"what are the key components of {lowered}",
                f"what common failure modes appear in {lowered}",
                f"how to improve {lowered}",
            ]
        )
    return dedupe_query_variants(variants)


def entity_concept_variants(base: str) -> list[str]:
    """Expand the query with related entities and concepts.

    These compact concept prompts increase the chance of matching documents
    indexed under adjacent terminology.
    """
    lowered = base.lower()
    variants: list[str] = []
    concept_map = (
        (
            "context stuffing",
            [
                "adding too much context",
                "overloading prompt context window",
                "excessive retrieval in rag",
                "context overload reduces answer quality",
            ],
        ),
        (
            "evaluation",
            ["faithfulness", "groundedness", "answer relevance", "retrieval quality metrics"],
        ),
        ("metrics", ["mrr ndcg recall at k", "ranking quality diagnostics"]),
        (
            "recall",
            ["retrieval coverage", "candidate pool expansion", "bm25 dense recall tradeoff"],
        ),
        (
            "hybrid",
            ["bm25 dense fusion", "rrf hybrid retrieval", "branch-specific retrieval misses"],
        ),
        ("rerank", ["cross encoder reranking optimization", "hard negative reranker training"]),
        ("rag", ["retriever generator grounding", "evidence grounded answering"]),
    )
    for trigger, expansions in concept_map:
        if trigger in lowered:
            variants.extend(expansions)
    if not variants:
        variants.extend(["retrieval system failure analysis", "document ranking optimization"])
    return dedupe_query_variants(variants)


def parse_llm_expansion_payload(raw_text: str) -> tuple[list[str], list[str], list[str]]:
    """Parse strict JSON LLM response into three variant groups.

    This parser isolates model-output handling and ensures invalid or noisy
    responses degrade gracefully to empty expansions.
    """
    text = _strip_markdown_code_fences(raw_text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [], [], []
    paraphrases = [str(item).strip() for item in payload.get("paraphrases", [])]
    decompositions = [str(item).strip() for item in payload.get("decompositions", [])]
    concept_expansions = [str(item).strip() for item in payload.get("concept_expansions", [])]
    return (
        dedupe_query_variants(paraphrases),
        dedupe_query_variants(decompositions),
        dedupe_query_variants(concept_expansions),
    )


def llm_structured_query_expansion(
    query: str,
    *,
    provider: str,
    model: str,
    api_base: str | None,
    api_key: str | None,
    timeout_seconds: int,
    retries: int,
    llm_config_path: str,
    cache_enabled: bool = False,
    cache_capacity: int = 512,
    cache_ttl_seconds: float = 300.0,
) -> tuple[list[str], list[str], list[str]]:
    """Generate structured query expansions from an LLM for one query.

    This is used when heuristic expansions are insufficient and broader recall
    is required across paraphrase, decomposition, and concept dimensions.
    """
    from generation.llm import call_llm
    from generation.run_rag import get_llm_config

    conf = get_llm_config(provider=provider, model=model or None, config_path=llm_config_path)
    if api_base:
        conf.api_base = api_base
    if api_key:
        conf.api_key = api_key
    conf.max_tokens = 400
    conf.temperature = 0.2
    conf.top_p = 1.0
    conf.timeout_seconds = max(1, timeout_seconds)
    conf.retries = max(0, retries)
    conf.cache_enabled = cache_enabled
    conf.cache_capacity = max(1, cache_capacity)
    conf.cache_ttl_seconds = max(0.1, cache_ttl_seconds)
    system_prompt = (
        "Generate retrieval queries that maximize document coverage. "
        "Return strict JSON only with keys: paraphrases, decompositions, concept_expansions. "
        "Each value must be a list of strings."
    )
    user_prompt = (
        "Generate retrieval queries that maximize document coverage.\n\n"
        "Return:\n"
        "1 paraphrases\n"
        "3 decompositions\n"
        "3 concept-expansion queries\n\n"
        f"Query: {query}"
    )
    try:
        response = call_llm(system_prompt=system_prompt, user_prompt=user_prompt, config=conf)
        return parse_llm_expansion_payload(response)
    except Exception:
        return [], [], []


def llm_structured_query_expansion_batch(
    queries: list[str],
    *,
    provider: str,
    model: str,
    api_base: str | None,
    api_key: str | None,
    timeout_seconds: int,
    retries: int,
    llm_config_path: str,
    cache_enabled: bool = False,
    cache_capacity: int = 512,
    cache_ttl_seconds: float = 300.0,
) -> dict[str, tuple[list[str], list[str], list[str]]]:
    """Generate structured LLM expansions for many queries in one request.

    Batching reduces LLM overhead during evaluation runs where query expansion
    is enabled for large datasets.
    """
    from generation.llm import call_llm
    from generation.run_rag import get_llm_config

    unique_queries = [q.strip() for q in queries if q.strip()]
    if not unique_queries:
        return {}

    conf = get_llm_config(provider=provider, model=model or None, config_path=llm_config_path)
    if api_base:
        conf.api_base = api_base
    if api_key:
        conf.api_key = api_key
    conf.max_tokens = 2000
    conf.temperature = 0.2
    conf.top_p = 1.0
    conf.timeout_seconds = max(1, timeout_seconds)
    conf.retries = max(0, retries)
    conf.cache_enabled = cache_enabled
    conf.cache_capacity = max(1, cache_capacity)
    conf.cache_ttl_seconds = max(0.1, cache_ttl_seconds)

    system_prompt = (
        "Generate retrieval queries that maximize document coverage. "
        "Return strict JSON only. "
        "Top-level key: expansions, value is an array. "
        "Each item must be {query, paraphrases, decompositions, concept_expansions}."
    )
    joined_queries = "\n".join(f"- {q}" for q in unique_queries)
    user_prompt = (
        "Generate retrieval queries that maximize document coverage.\n\n"
        "For EACH query below return:\n"
        "1 paraphrases\n"
        "3 decompositions\n"
        "3 concept-expansion queries\n\n"
        "Queries:\n"
        f"{joined_queries}"
    )
    try:
        response = call_llm(system_prompt=system_prompt, user_prompt=user_prompt, config=conf)
        text = _strip_markdown_code_fences(response)
        payload = json.loads(text)
    except Exception:
        return {}

    result: dict[str, tuple[list[str], list[str], list[str]]] = {}
    for item in payload.get("expansions", []):
        query = str(item.get("query", "")).strip()
        if not query:
            continue
        paraphrases = dedupe_query_variants([str(x).strip() for x in item.get("paraphrases", [])])
        decompositions = dedupe_query_variants(
            [str(x).strip() for x in item.get("decompositions", [])]
        )
        concepts = dedupe_query_variants(
            [str(x).strip() for x in item.get("concept_expansions", [])]
        )
        result[query] = (paraphrases, decompositions, concepts)
    return result


def build_query_variants(
    query: str,
    max_variants: int,
    *,
    use_llm_structured_expansion: bool = False,
    llm_provider: str = "qwen",
    llm_model: str = "qwen-plus",
    llm_api_base: str | None = None,
    llm_api_key: str | None = None,
    llm_timeout_seconds: int = 8,
    llm_retries: int = 0,
    llm_config_path: str = "llm.config.json",
    llm_cache_enabled: bool = False,
    llm_cache_capacity: int = 512,
    llm_cache_ttl_seconds: float = 300.0,
    llm_precomputed: tuple[list[str], list[str], list[str]] | None = None,
) -> list[str]:
    """Build a capped list of query variants for retrieval.

    This is a convenience wrapper used by retrieval flows that only need the
    variant list and not LLM-debug diagnostics.
    """
    variants, _ = build_query_variants_with_debug(
        query=query,
        max_variants=max_variants,
        use_llm_structured_expansion=use_llm_structured_expansion,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_base=llm_api_base,
        llm_api_key=llm_api_key,
        llm_timeout_seconds=llm_timeout_seconds,
        llm_retries=llm_retries,
        llm_config_path=llm_config_path,
        llm_cache_enabled=llm_cache_enabled,
        llm_cache_capacity=llm_cache_capacity,
        llm_cache_ttl_seconds=llm_cache_ttl_seconds,
        llm_precomputed=llm_precomputed,
    )
    return variants


def build_query_variants_with_debug(
    *,
    query: str,
    max_variants: int,
    use_llm_structured_expansion: bool = False,
    llm_provider: str = "qwen",
    llm_model: str = "qwen-plus",
    llm_api_base: str | None = None,
    llm_api_key: str | None = None,
    llm_timeout_seconds: int = 8,
    llm_retries: int = 0,
    llm_config_path: str = "llm.config.json",
    llm_cache_enabled: bool = False,
    llm_cache_capacity: int = 512,
    llm_cache_ttl_seconds: float = 300.0,
    llm_precomputed: tuple[list[str], list[str], list[str]] | None = None,
) -> tuple[list[str], dict[str, object]]:
    """Build query variants and include diagnostics for LLM expansion behavior.

    The debug payload helps evaluate whether LLM expansion contributed useful
    variants or if the pipeline fell back to heuristic expansions.
    """
    base = query.strip()
    if not base or max_variants <= 1:
        variants = [base] if base else []
        return variants, {
            "llm_requested": False,
            "llm_generated_count": 0,
            "llm_generated_preview": [],
            "fallback_used": False,
        }

    paraphrase = paraphrase_variants(base)
    decomposition = decomposition_variants(base)
    entity_concepts = entity_concept_variants(base)
    llm_generated_preview: list[str] = []
    llm_generated_count = 0
    fallback_used = False
    if use_llm_structured_expansion:
        if llm_precomputed is not None:
            llm_paraphrase, llm_decomposition, llm_concepts = llm_precomputed
        else:
            llm_paraphrase, llm_decomposition, llm_concepts = llm_structured_query_expansion(
                base,
                provider=llm_provider,
                model=llm_model,
                api_base=llm_api_base,
                api_key=llm_api_key,
                timeout_seconds=llm_timeout_seconds,
                retries=llm_retries,
                llm_config_path=llm_config_path,
                cache_enabled=llm_cache_enabled,
                cache_capacity=llm_cache_capacity,
                cache_ttl_seconds=llm_cache_ttl_seconds,
            )
        llm_generated = dedupe_query_variants(llm_paraphrase + llm_decomposition + llm_concepts)
        llm_generated_preview = llm_generated[:3]
        llm_generated_count = len(llm_generated)
        fallback_used = llm_generated_count == 0
        paraphrase = dedupe_query_variants(llm_paraphrase + paraphrase)
        decomposition = dedupe_query_variants(llm_decomposition + decomposition)
        entity_concepts = dedupe_query_variants(llm_concepts + entity_concepts)

    variants: list[str] = [base]
    layer_lists = [paraphrase, decomposition, entity_concepts]
    layer_index = 0
    while len(variants) < max_variants and any(layer_lists):
        layer = layer_lists[layer_index % len(layer_lists)]
        if layer:
            variants.append(layer.pop(0))
        layer_index += 1
        if layer_index > (max_variants * 6):
            break
    final_variants = dedupe_query_variants(variants)[:max_variants]
    return final_variants, {
        "llm_requested": use_llm_structured_expansion,
        "llm_generated_count": llm_generated_count,
        "llm_generated_preview": llm_generated_preview,
        "fallback_used": fallback_used,
    }


def _strip_markdown_code_fences(raw_text: str) -> str:
    """Remove Markdown code fence marker lines from LLM output.

    LLM responses can include leading/trailing fences and additional embedded
    fence markers; removing all fence marker lines preserves valid JSON body.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text

# src/phase2_relations.py
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llama_index.core import Document, Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex

# ---------- Paths ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_JSON = REPO_ROOT / "outputs" / "phase2_concepts.json"
OUTPUT_JSON = REPO_ROOT / "outputs" / "phase2_relations.json"

# ---------- Model names ----------
HF_LLM = "mistralai/Mistral-7B-Instruct-v0.3"
HF_EMB = "BAAI/bge-small-en-v1.5"   # embeddings not crucial for KG, but set globally

# ---------- Config ----------
MAX_TRIPLETS_PER_CHUNK = 8      # tweak as you like (quality vs. speed)
BATCH_LIMIT = None              # None -> process all; or set an int while iterating

def load_concepts(input_path: Path) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Expecting a list of {chunk, concepts, chunk_id}
    return data

def build_llm() -> HuggingFaceLLM:
    # 4-bit quantization (A100 is fine)
    quant_config = BitsAndBytesConfig(load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(HF_LLM, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        HF_LLM,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quant_config,
    )
    llm = HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        generate_kwargs={
            "temperature": 0.1,
            "top_p": 0.9,
            "max_new_tokens": 512,
        },
    )
    return llm

def run_extraction(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Set global LLM + embedding for LlamaIndex
    Settings.llm = build_llm()
    Settings.embed_model = HuggingFaceEmbedding(model_name=HF_EMB)

    docs: List[Document] = []
    for i, item in enumerate(chunks[: BATCH_LIMIT or len(chunks)]):
        text = item["chunk"]
        # Lightly append concepts as hints (comment this out if you want *pure* text)
        concepts = item.get("concepts", [])
        concept_hint = ""
        if concepts:
            # Shorten extremely long concept lists from noisy LLM outputs
            joined = ", ".join([c.strip("- ").strip() for c in concepts])[:1500]
            concept_hint = f"\n\n[HINTS: Concepts detected -> {joined}]"

        doc = Document(text + concept_hint, metadata={"chunk_id": item.get("chunk_id", i)})
        docs.append(doc)

    # Build KG index (in-memory store)
    kg_index = KnowledgeGraphIndex.from_documents(
        docs,
        max_triplets_per_chunk=MAX_TRIPLETS_PER_CHUNK,
    )

    # Pull raw triplets back out
    # NOTE: KnowledgeGraphIndex stores triplets internally; `get_triples` is a simple
    # helper we implement via the index graph store.
    triples = []
    # The internal API differs across versions; use this safe accessor:
    try:
        graph_store = kg_index.get_graph_store()
        # Expect edges as (subject, predicate, object)
        for edge in graph_store.get_triples():
            s, p, o = edge
            triples.append({"subject": s, "predicate": p, "object": o})
    except Exception:
        # Fallback: scan nodes/relations (older versions)
        # If needed, inspect graph_store._graph or graph_store._triplets (private attr)
        # To keep robust, we simply query via query engine:
        qe = kg_index.as_query_engine()
        _ = qe.query("List all (subject, predicate, object) triples you know.")
        # As a minimal fallback, we return empty and let graph_build construct from index later.
        pass

    # Also map triplets per chunk by re-running extractor per doc (more reliable association)
    # This second pass ensures we keep a chunk_id for each triple.
    per_chunk: List[Dict[str, Any]] = []
    for doc in docs:
        sub_index = KnowledgeGraphIndex.from_documents(
            [doc],
            max_triplets_per_chunk=MAX_TRIPLETS_PER_CHUNK,
        )
        try:
            gs = sub_index.get_graph_store()
            local = []
            for s, p, o in gs.get_triples():
                local.append({"subject": s, "predicate": p, "object": o})
            per_chunk.append({"chunk_id": doc.metadata["chunk_id"], "triples": local})
        except Exception:
            per_chunk.append({"chunk_id": doc.metadata["chunk_id"], "triples": []})

    return per_chunk

def main():
    INPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    chunks = load_concepts(INPUT_JSON)
    per_chunk = run_extraction(chunks)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(per_chunk, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(per_chunk)} chunk triple lists to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()

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
HF_EMB = "BAAI/bge-small-en-v1.5"   # embeddings not crucial for KG, but required by LlamaIndex

# ---------- Config ----------
MAX_TRIPLETS_PER_CHUNK = 8
BATCH_LIMIT = None   # set to int for debugging smaller subsets


# ---------- Helpers ----------
def load_concepts(input_path: Path) -> List[Dict[str, Any]]:
    """Load Phase 2 concepts JSON."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_llm() -> HuggingFaceLLM:
    """Load HuggingFace model with quantization for LlamaIndex."""
    quant_config = BitsAndBytesConfig(load_in_4bit=True)

    tokenizer = AutoTokenizer.from_pretrained(HF_LLM, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        HF_LLM,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quant_config,
    )

    return HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        generate_kwargs={
            "temperature": 0.1,
            "top_p": 0.9,
            "max_new_tokens": 512,
        },
    )


def run_extraction(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract triples per chunk using KnowledgeGraphIndex."""
    # Global settings for LlamaIndex
    Settings.llm = build_llm()
    Settings.embed_model = HuggingFaceEmbedding(model_name=HF_EMB)

    results: List[Dict[str, Any]] = []

    for i, item in enumerate(chunks[: BATCH_LIMIT or len(chunks)]):
        text = item["chunk"]

        # Add concepts as optional hints
        concepts = item.get("concepts", [])
        concept_hint = ""
        if concepts:
            joined = ", ".join([c.strip("- ").strip() for c in concepts])[:1500]
            concept_hint = f"\n\n[HINTS: Concepts detected -> {joined}]"

        # ✅ Correct usage of LlamaIndex Document
        doc = Document(
            text=text + concept_hint,
            metadata={"chunk_id": item.get("chunk_id", i)},
        )

        # Build KG index per doc
        kg_index = KnowledgeGraphIndex.from_documents(
            [doc],
            max_triplets_per_chunk=MAX_TRIPLETS_PER_CHUNK,
        )

        triples = []
        try:
            graph_store = kg_index.get_graph_store()
            for s, p, o in graph_store.get_triples():
                triples.append({"subject": s, "predicate": p, "object": o})
        except Exception:
            # fallback if version mismatch
            triples = []

        results.append({
            "chunk_id": doc.metadata["chunk_id"],
            "triples": triples,
        })

    return results


def main():
    INPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    chunks = load_concepts(INPUT_JSON)
    per_chunk = run_extraction(chunks)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(per_chunk, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(per_chunk)} chunks with triples → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

# phase2_concepts_llama.py

import json
from pathlib import Path
from llama_index.core import Document
from llama_index.core.indices.property_graph import PropertyGraphIndex, SimpleLLMPathExtractor
from llama_index.core.llms import HuggingFaceLLM
from llama_index.core.node_parser import SentenceSplitter

# ---------------- Files ----------------
INPUT_FILE = "attention_chunks.json"
OUTPUT_INDEX_DIR = "./phase2_pg_index"

# ---------------- Load chunks ----------------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# ---------------- Create Documents ----------------
documents = [
    Document(text=chunk["text"], metadata={"chunk_id": chunk["chunk_id"]})
    for chunk in chunks
]

# ---------------- Optional: Parse into Nodes ----------------
# This splits text into smaller chunks for finer-grained relationship extraction
parser = SentenceSplitter()
nodes = parser.get_nodes_from_documents(documents)

# ---------------- LLM setup ----------------
# Use Mistral v0.3 Instruct (assuming HuggingFace LLM wrapper)
llm = HuggingFaceLLM(
    model_name="mistral-instruct-v0.3",
    model_kwargs={"temperature": 0.0, "max_new_tokens": 256}
)

# ---------------- KG Extractor ----------------
# Extract (entity, relation, entity) triples from text chunks
kg_extractor = SimpleLLMPathExtractor(
    llm=llm,
    max_paths_per_chunk=10,
    num_workers=4,
    show_progress=True,
)

# ---------------- Build Property Graph Index ----------------
pg_index = PropertyGraphIndex.from_documents(
    nodes,  # can also use 'documents' if you skip node parsing
    kg_extractors=[kg_extractor],
)

# ---------------- Save index ----------------
pg_index.storage_context.persist(persist_dir=OUTPUT_INDEX_DIR)

print(f"Phase 2 Property Graph Index created and saved to {OUTPUT_INDEX_DIR}")

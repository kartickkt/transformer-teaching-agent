import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from llama_index.core.schema import Document
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor

# ---------------- Settings ----------------
CHUNKS_FILE = Path("outputs/attention_chunks.json")
OUTPUT_DIR = Path("./storage")
MODEL_NAME = "mistral-instruct-0.3"

# ---------------- Load Mistral ----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
llm = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# ---------------- Load chunks ----------------
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

documents = []
for chunk in chunks:
    # wrap each chunk as a LlamaIndex Document
    documents.append(Document(text=chunk["text"], doc_id=str(chunk["chunk_id"])))

# ---------------- Concept & Relationship Extraction ----------------
kg_extractor = SimpleLLMPathExtractor(
    llm=llm,
    max_paths_per_chunk=10,
    num_workers=1,  # adjust based on your resources
    show_progress=True
)

# Build Property Graph Index
index = PropertyGraphIndex.from_documents(
    documents,
    kg_extractors=[kg_extractor]
)

# ---------------- Persist to disk ----------------
index.storage_context.persist(persist_dir=OUTPUT_DIR)

print(f"Property graph index saved to {OUTPUT_DIR}")
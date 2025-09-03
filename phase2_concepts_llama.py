# src/phase2_relations.py
import json
from pathlib import Path
from llama_index.core import PropertyGraphIndex, StorageContext
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.core.schema import TextNode

# Optional: your LLM setup
from llama_index.langchain_helpers.chain_wrapper import LLMChain
from langchain.chat_models import ChatOpenAI

# -------------------- Settings --------------------
CHUNKS_FILE = Path("outputs/attention_chunks.json")
OUTPUT_DIR = Path("./storage")
MAX_PATHS_PER_CHUNK = 10

# -------------------- Load chunks --------------------
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Convert chunks to LlamaIndex TextNode objects
documents = [TextNode(id_=str(c["chunk_id"]), text=c["text"]) for c in chunks]

# -------------------- Setup LLM --------------------
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")  # adjust as needed

kg_extractor = SimpleLLMPathExtractor(
    llm=llm,
    max_paths_per_chunk=MAX_PATHS_PER_CHUNK,
    num_workers=4,
    show_progress=True,
)

# -------------------- Build Property Graph Index --------------------
index = PropertyGraphIndex.from_documents(
    documents,
    kg_extractors=[kg_extractor],
)

# -------------------- Persist Index --------------------
index.storage_context.persist(persist_dir=OUTPUT_DIR)
print(f"Property graph index saved to {OUTPUT_DIR}")

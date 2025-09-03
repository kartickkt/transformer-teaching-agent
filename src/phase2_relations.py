# phase2_relations.py
import json
from pathlib import Path
from tqdm import tqdm

from llama_index import LLMPredictor, ServiceContext, Document
from llama_index.indices.knowledge_graph.base import KnowledgeGraphIndex
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ---------------- Paths ----------------
ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT_DIR / "outputs/phase2_concepts.json"
OUTPUT_FILE = ROOT_DIR / "outputs/phase2_relationships.json"

# ---------------- Load concepts ----------------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    concept_chunks = json.load(f)

documents = []
for chunk in concept_chunks:
    text = chunk.get("chunk", "")
    documents.append(Document(text=text))

# ---------------- Setup LLM (Mistral-7B) ----------------
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0")
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

llm_predictor = LLMPredictor(model=model, tokenizer=tokenizer)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# ---------------- Build Knowledge Graph ----------------
kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    service_context=service_context
)

# ---------------- Extract relationships per chunk ----------------
per_chunk_relationships = []
for chunk in tqdm(concept_chunks):
    doc = Document(text=chunk.get("chunk", ""))
    relations = kg_index.extract_triplets_from_document(doc)
    per_chunk_relationships.append({
        "chunk_id": chunk["chunk_id"],
        "relationships": relations
    })

# ---------------- Save output ----------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(per_chunk_relationships, f, indent=2)

print(f"✅ Relationships saved to {OUTPUT_FILE}")

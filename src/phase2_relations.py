import json
from pathlib import Path
from llama_index.core import KnowledgeGraphIndex, Document, ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM

# ---------------- Settings ----------------
CONCEPTS_FILE = "phase2_concepts.json"
OUTPUT_FILE = "phase2_relationships.json"

# ---------------- Hugging Face LLM ----------------
# Using Mistral (or any HF model you have access to)
hf_llm = HuggingFaceLLM(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    max_new_tokens=512,
    device="cuda"  # uses your A100
)

service_context = ServiceContext.from_defaults(llm=hf_llm)

# ---------------- Load Concepts ----------------
with open(CONCEPTS_FILE, "r") as f:
    concept_chunks = json.load(f)

# Convert each chunk to a Document
documents = []
for chunk in concept_chunks:
    text = "\n".join(chunk["concepts"])  # join the list of concepts into a single text
    documents.append(Document(text=text))

# ---------------- Build Knowledge Graph ----------------
kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    service_context=service_context
)

# ---------------- Extract Relationships ----------------
relationships_per_chunk = []

for doc in documents:
    # Returns a dictionary with extracted triplets
    triplets = kg_index.extract_triplets_from_text(doc.text)
    relationships_per_chunk.append({
        "text": doc.text,
        "triplets": triplets
    })

# ---------------- Save ----------------
with open(OUTPUT_FILE, "w") as f:
    json.dump(relationships_per_chunk, f, indent=2)

print(f"✅ Relationships extracted and saved to {OUTPUT_FILE}")

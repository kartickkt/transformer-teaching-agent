# phase2_concepts_llama.py

import json
from pathlib import Path
from llama_index import Document
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.indices.property_graph import PropertyGraphIndex, SimpleLLMPathExtractor
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ----------------- Settings -----------------
INPUT_FILE = "outputs/attention_chunks.json"
OUTPUT_FILE = "outputs/phase2_concepts_llama.json"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"  # or any other LLM

# ----------------- Load chunks -----------------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# ----------------- Setup LLM -----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype="auto"
)
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)

# Wrap LLM in a callable for LlamaIndex
class LLMPipelineWrapper:
    def __call__(self, prompt, **kwargs):
        result = llm_pipeline(prompt, max_new_tokens=256)[0]["generated_text"]
        return result

llm = LLMPipelineWrapper()

# ----------------- Create Documents -----------------
documents = [Document(text=chunk["text"], doc_id=str(chunk["chunk_id"])) for chunk in chunks]

# ----------------- Create Extractor -----------------
kg_extractor = SimpleLLMPathExtractor(
    llm=llm,
    max_paths_per_chunk=20,
    num_workers=4
)

# ----------------- Extract concepts -----------------
extracted_data = []

for doc in documents:
    paths = kg_extractor([doc])
    # paths is a list of dicts containing entities/relations
    concepts = [p["object"] for p in paths[0].metadata.get("kg_relations", []) if "object" in p]
    extracted_data.append({
        "chunk_id": int(doc.doc_id),
        "chunk": doc.text,
        "concepts": concepts
    })

# ----------------- Save output -----------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(extracted_data, f, ensure_ascii=False, indent=2)

print(f"✅ Concepts extracted and saved to {OUTPUT_FILE}")

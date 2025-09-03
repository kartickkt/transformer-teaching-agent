import json
from llama_index.core import KnowledgeGraphIndex, Document, Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ---------------- Settings ----------------
CONCEPTS_FILE = "outputs/phase2_concepts.json"
OUTPUT_FILE = "outputs/phase2_relationships.json"

# ---------------- Hugging Face LLM ----------------
hf_llm = HuggingFaceLLM(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
    max_new_tokens=512,
    device_map="auto",
    model_kwargs={"torch_dtype": "auto"}
)

# ---------------- Hugging Face Embeddings ----------------
# You can pick any lightweight embedding model (sentence-transformers recommended)
hf_embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Register globally (replaces ServiceContext)
Settings.llm = hf_llm
Settings.embed_model = hf_embed_model   # ✅ prevents fallback to OpenAI

# ---------------- Load Concepts ----------------
with open(CONCEPTS_FILE, "r") as f:
    concept_chunks = json.load(f)

# Convert each chunk to a Document
documents = []
for chunk in concept_chunks:
    text = "\n".join(chunk["concepts"])  # join the list of concepts into a single text
    documents.append(Document(text=text))

# ---------------- Build Knowledge Graph ----------------
kg_index = KnowledgeGraphIndex.from_documents(documents)

# ---------------- Extract Relationships ----------------
relationships_per_chunk = []

for doc in documents:
    try:
        triplets = kg_index.extract_triplets_from_text(doc.text)
    except Exception as e:
        triplets = []
        print(f"⚠️ Could not extract triplets for: {doc.text[:50]}... Error: {e}")
    
    relationships_per_chunk.append({
        "text": doc.text,
        "triplets": triplets
    })

# ---------------- Save ----------------
with open(OUTPUT_FILE, "w") as f:
    json.dump(relationships_per_chunk, f, indent=2)

print(f"✅ Relationships extracted and saved to {OUTPUT_FILE}")

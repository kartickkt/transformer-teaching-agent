# src/phase2_relations.py

import json
from pathlib import Path
from tqdm import tqdm
from llama_index import KnowledgeGraphIndex, LLMPredictor, ServiceContext, Document
from llama_index.llms import HuggingFaceLLM

# ---------------- Path setup ----------------
repo_root = Path(__file__).resolve().parents[1]  # adjust if needed
input_file = repo_root / "outputs/phase2_concepts.json"
output_file = repo_root / "outputs/phase2_relations.json"

# ---------------- Helper functions ----------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ---------------- Main extraction function ----------------
def run_extraction(concepts_data):
    """
    Build Knowledge Graph from extracted concepts.
    Each concept entry is joined into a document for the KG.
    """
    # Convert concepts_data to Document objects
    documents = []
    for entry in concepts_data:
        # Some entries may be lists of strings inside "concepts"
        if not entry["concepts"]:
            continue
        doc_text = "\n".join(entry["concepts"])
        documents.append(Document(text=doc_text))

    # Initialize HuggingFace LLM
    llm = HuggingFaceLLM(
        model_name="tiiuae/falcon-7b-instruct",
        max_new_tokens=512,
        temperature=0.0,
        device_map="auto"  # GPU if available
    )

    # Create service context for KG
    service_context = ServiceContext.from_defaults(llm_predictor=LLMPredictor(llm=llm))

    # Build Knowledge Graph Index
    print("Building Knowledge Graph...")
    kg_index = KnowledgeGraphIndex.from_documents(
        documents,
        service_context=service_context
    )

    # Extract relationships per chunk
    per_chunk = []
    for doc in tqdm(documents):
        triplets = kg_index.get_triplets(doc.text)
        per_chunk.append({"text": doc.text, "triplets": triplets})

    return per_chunk

# ---------------- Main ----------------
def main():
    concepts_data = load_json(input_file)
    relations = run_extraction(concepts_data)
    save_json(relations, output_file)
    print(f"Saved {len(relations)} entries to {output_file}")

if __name__ == "__main__":
    main()

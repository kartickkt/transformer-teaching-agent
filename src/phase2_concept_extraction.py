import json
import os
from pathlib import Path
import sys
from tqdm import tqdm

# ---------------- Path setup ----------------
repo_root = Path(__file__).resolve().parents[1]  # go up one level from src/
sys.path.append(str(repo_root))

from src.llm_utils import call_llm  # assumes you have a wrapper for your LLM calls
from src.utils import load_json, save_json

# ---------------- Files ----------------
INPUT_CHUNKS_FILE = repo_root / "outputs" / "attention_chunks.json"
OUTPUT_CONCEPTS_FILE = repo_root / "outputs" / "concepts.json"
OUTPUT_GRAPH_FILE = repo_root / "outputs" / "concept_graph.json"

# ---------------- Utility functions ----------------
def clean_chunk_text(text):
    """
    Remove section numbers, headings, and empty strings from a chunk.
    """
    if not text or not text.strip():
        return None
    # Remove common patterns like "1. Introduction", "2.1 Something"
    import re
    text = re.sub(r'^\d+(\.\d+)*\s+', '', text.strip())
    return text if text else None

def extract_concepts_from_chunk(chunk_text):
    """
    Use an LLM to extract key concepts from a text chunk.
    Returns a list of concepts.
    """
    if not chunk_text:
        return []

    prompt = f"""
    Extract the key concepts from the following text. 
    Only return a JSON list of concise concept names. Avoid hallucinations.
    Text:
    \"\"\"{chunk_text}\"\"\"
    """

    response = call_llm(prompt)  # your wrapper should return text
    try:
        concepts = json.loads(response)
        if isinstance(concepts, list):
            return [c.strip() for c in concepts if c.strip()]
    except json.JSONDecodeError:
        print("LLM output could not be parsed as JSON:", response)
    return []

def build_concept_graph(concepts_per_chunk):
    """
    Build a simple knowledge graph with edges:
    - Prerequisite (A requires B)
    - Related (A related B)
    For starter, we just link consecutive concepts as related.
    """
    edges = []
    all_concepts = set()
    
    for chunk_concepts in concepts_per_chunk:
        for c in chunk_concepts:
            all_concepts.add(c)
        # Simple heuristic: link consecutive concepts in a chunk as related
        for i in range(len(chunk_concepts) - 1):
            edges.append({
                "source": chunk_concepts[i],
                "target": chunk_concepts[i+1],
                "relation": "related"
            })
    return list(all_concepts), edges

# ---------------- Main pipeline ----------------
def main():
    print("Phase 2: Concept Extraction & Knowledge Graph Construction")

    # Load Phase 1 chunks
    chunks = load_json(INPUT_CHUNKS_FILE)
    print(f"Loaded {len(chunks)} chunks from {INPUT_CHUNKS_FILE}")

    # Process chunks
    concepts_per_chunk = []
    for idx, chunk in enumerate(tqdm(chunks, desc="Extracting concepts")):
        cleaned_text = clean_chunk_text(chunk.get("text", ""))
        if not cleaned_text:
            continue
        concepts = extract_concepts_from_chunk(cleaned_text)
        if concepts:
            concepts_per_chunk.append(concepts)

    # Save concepts per chunk
    save_json(concepts_per_chunk, OUTPUT_CONCEPTS_FILE)
    print(f"Saved extracted concepts to {OUTPUT_CONCEPTS_FILE}")

    # Build concept graph
    all_concepts, concept_edges = build_concept_graph(concepts_per_chunk)
    concept_graph = {
        "nodes": [{"id": c} for c in all_concepts],
        "edges": concept_edges
    }
    save_json(concept_graph, OUTPUT_GRAPH_FILE)
    print(f"Saved concept graph to {OUTPUT_GRAPH_FILE}")

if __name__ == "__main__":
    main()

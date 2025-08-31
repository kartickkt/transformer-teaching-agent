import json
from pathlib import Path
from tqdm import tqdm

# ----------------- LangChain & LangGraph imports -----------------
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline  # Use Hugging Face model
from transformers import pipeline
from langgraph import Graph, Node, Edge

# ----------------- Local utils -----------------
from src.utils import load_json, save_json

# ----------------- Paths -----------------
repo_root = Path(__file__).resolve().parents[1]
INPUT_CHUNKS_FILE = repo_root / "outputs" / "attention_chunks.json"
OUTPUT_CONCEPTS_FILE = repo_root / "outputs" / "concepts.json"
OUTPUT_GRAPH_FILE = repo_root / "outputs" / "concept_graph.json"

# ----------------- Helper Functions -----------------
def clean_chunk_text(text):
    """Remove empty strings and section numbers."""
    import re
    if not text or not text.strip():
        return None
    text = re.sub(r'^\d+(\.\d+)*\s+', '', text.strip())
    return text if text else None

def extract_concepts_langchain(chunk_text, llm_chain):
    """Use LangChain + Mistral to extract key concepts."""
    if not chunk_text:
        return []
    response = llm_chain.run({"chunk_text": chunk_text})
    try:
        concepts = json.loads(response)
        if isinstance(concepts, list):
            return [c.strip() for c in concepts if c.strip()]
    except json.JSONDecodeError:
        print("LLM output could not be parsed as JSON:", response)
    return []

def build_langgraph(concepts_per_chunk):
    """Build a LangGraph knowledge graph from concepts."""
    g = Graph()
    all_concepts = set(c for chunk in concepts_per_chunk for c in chunk)
    for c in all_concepts:
        g.add_node(Node(id=c, label=c))
    for chunk in concepts_per_chunk:
        for i in range(len(chunk)-1):
            g.add_edge(Edge(source=chunk[i], target=chunk[i+1], relation="related"))
    return g

# ----------------- Main Pipeline -----------------
def main():
    print("Phase 2: Concept Extraction & Knowledge Graph Construction (Mistral)")

    chunks = load_json(INPUT_CHUNKS_FILE)
    print(f"Loaded {len(chunks)} chunks")

    # ----------------- Setup Mistral with LangChain -----------------
    # Hugging Face pipeline
    hf_pipe = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-v0.1",  # or your local Mistral path
        max_new_tokens=128,
        temperature=0.0
    )
    llm = HuggingFacePipeline(pipeline=hf_pipe)

    # LangChain prompt
    prompt_template = """
    Extract the key concepts from the following text.
    Only return a JSON list of concise concept names. Avoid hallucinations.
    Text:
    {chunk_text}
    """
    prompt = PromptTemplate(input_variables=["chunk_text"], template=prompt_template)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # ----------------- Extract concepts -----------------
    concepts_per_chunk = []
    for chunk in tqdm(chunks, desc="Extracting concepts"):
        text = clean_chunk_text(chunk.get("text", ""))
        concepts = extract_concepts_langchain(text, llm_chain)
        if concepts:
            concepts_per_chunk.append(concepts)

    save_json(concepts_per_chunk, OUTPUT_CONCEPTS_FILE)
    print(f"Saved concepts to {OUTPUT_CONCEPTS_FILE}")

    # ----------------- Build LangGraph -----------------
    graph = build_langgraph(concepts_per_chunk)
    save_json(graph.to_dict(), OUTPUT_GRAPH_FILE)
    print(f"Saved concept graph to {OUTPUT_GRAPH_FILE}")

if __name__ == "__main__":
    main()

# src/phase2_concept_extraction.py

import json
from pathlib import Path
from tqdm import tqdm
import re

# ----------------- LangChain & LangGraph -----------------
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from langgraph import Graph, Node, Edge

# ----------------- Transformers -----------------
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ----------------- Paths -----------------
repo_root = Path(__file__).resolve().parents[1]
INPUT_FILE = repo_root / "outputs" / "attention_chunks.json"
OUTPUT_CONCEPTS_FILE = repo_root / "outputs" / "concepts.json"
OUTPUT_GRAPH_FILE = repo_root / "outputs" / "concept_graph.json"

# ----------------- Helper Functions -----------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def clean_chunk_text(text):
    """Remove empty strings and section numbers."""
    if not text or not text.strip():
        return None
    text = re.sub(r'^\d+(\.\d+)*\s+', '', text.strip())
    return text if text else None

def extract_concepts(chunk_text, llm_chain):
    """Extract concepts from a chunk using the LangChain + local Mistral model."""
    if not chunk_text:
        return []
    response = llm_chain.run({"chunk_text": chunk_text})
    try:
        concepts = json.loads(response)
        if isinstance(concepts, list):
            return [c.strip() for c in concepts if c.strip()]
    except json.JSONDecodeError:
        print("Warning: LLM output could not be parsed as JSON:", response)
    return []

def build_graph(concepts_per_chunk):
    """Build LangGraph from extracted concepts."""
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
    print("Phase 2: Concept Extraction & Knowledge Graph (Local Mistral on A100)")

    # ----------------- Load chunks -----------------
    chunks = load_json(INPUT_FILE)
    print(f"Loaded {len(chunks)} chunks")

    # ----------------- Load Mistral locally -----------------
    print("Loading Mistral-7B model (4-bit, GPU)... This may take a minute")
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,          # 4-bit quantization for A100
        device_map="auto",          # automatically place on GPU
        torch_dtype=torch.float16
    )

    hf_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        temperature=0.0,
        device=0  # first GPU (A100)
    )

    llm = HuggingFacePipeline(pipeline=hf_pipe)

    # ----------------- LangChain prompt -----------------
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
        concepts = extract_concepts(text, llm_chain)
        if concepts:
            concepts_per_chunk.append(concepts)

    save_json(concepts_per_chunk, OUTPUT_CONCEPTS_FILE)
    print(f"Saved concepts to {OUTPUT_CONCEPTS_FILE}")

    # ----------------- Build concept graph -----------------
    graph = build_graph(concepts_per_chunk)
    save_json(graph.to_dict(), OUTPUT_GRAPH_FILE)
    print(f"Saved concept graph to {OUTPUT_GRAPH_FILE}")

if __name__ == "__main__":
    main()

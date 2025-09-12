# phase5_build_concept_graph.py

import json
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- Config ---
INPUT_CONCEPTS = "outputs/final_concepts.json"
INPUT_EMBEDDINGS = "outputs/final_embeddings.npy"
OUTPUT_GRAPH = "outputs/final_concept_graph.gexf"

SIMILARITY_THRESHOLD = 0.75  # tweak as needed

def main():
    # Load concepts (dicts)
    with open(INPUT_CONCEPTS, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract text only
    concepts = [c["text"] for c in data]

    # Load embeddings (NumPy array, same order)
    embeddings = np.load(INPUT_EMBEDDINGS)

    if len(concepts) != embeddings.shape[0]:
        raise ValueError(f"Mismatch: {len(concepts)} concepts vs {embeddings.shape[0]} embeddings")

    # Prepare graph
    G = nx.Graph()

    # Add nodes (concept strings)
    for c in concepts:
        G.add_node(c)

    # Build edges using cosine similarity
    sim_matrix = cosine_similarity(embeddings)

    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            if sim_matrix[i, j] >= SIMILARITY_THRESHOLD:
                G.add_edge(
                    concepts[i],
                    concepts[j],
                    weight=float(sim_matrix[i, j])
                )

    # Save graph
    nx.write_gexf(G, OUTPUT_GRAPH)
    print(f"✅ Graph built and saved to {OUTPUT_GRAPH}")
    print(f"Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")

if __name__ == "__main__":
    main()

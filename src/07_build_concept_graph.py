import json
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Paths
embeddings_path = Path("temp/06_final_embeddings.json")
json_graph_path = Path("temp/07_concept_graph.json")
gexf_graph_path = Path("temp/07_concept_graph.gexf")

# Load embeddings
with open(embeddings_path, "r") as f:
    embeddings_data = json.load(f)

concepts = list(embeddings_data.keys())
embeddings = np.array(list(embeddings_data.values()))

print(f"Loaded {len(concepts)} concepts with embeddings")

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(embeddings)

# Build edge list with threshold
threshold = 0.7  # adjust as needed
edges = []
for i in range(len(concepts)):
    for j in range(i + 1, len(concepts)):
        weight = similarity_matrix[i, j]
        if weight >= threshold:
            edges.append({
                "source": concepts[i],
                "target": concepts[j],
                "weight": float(weight)
            })

# Build JSON graph
graph = {
    "nodes": [{"id": c} for c in concepts],
    "edges": edges
}

# Save JSON graph
with open(json_graph_path, "w") as f:
    json.dump(graph, f, indent=2)

print(f"✅ JSON graph saved to {json_graph_path}")
print(f"Nodes: {len(graph['nodes'])}, Edges: {len(graph['edges'])}")

# Build NetworkX graph for visualization
G = nx.Graph()
for node in concepts:
    # label = concept itself
    G.add_node(node, label=node)

for edge in edges:
    G.add_edge(edge["source"], edge["target"], weight=edge["weight"])

# Save to GEXF for Gephi
nx.write_gexf(G, gexf_graph_path)
print(f"✅ GEXF graph saved to {gexf_graph_path} (for visualization only)")

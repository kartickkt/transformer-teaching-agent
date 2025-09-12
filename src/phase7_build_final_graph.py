import os
import json
import networkx as nx

# Paths
concepts_path = "outputs/final_concepts.json"
relationships_path = "outputs/final_relationships.json"
graph_path = "outputs/final_concept_graph_with_relationships.gexf"

# Load concepts
with open(concepts_path, "r", encoding="utf-8") as f:
    concepts = json.load(f)

# Load relationships
with open(relationships_path, "r", encoding="utf-8") as f:
    relationships = json.load(f)

# Create directed graph
G = nx.DiGraph()

# Add nodes (concepts)
for concept in concepts:
    G.add_node(concept, label=concept)  # label metadata

# Add edges (relationships)
for rel in relationships:
    source = rel.get("source")
    target = rel.get("target")
    rel_type = rel.get("type", "related_to")

    # Add edge with type as label
    if source in G and target:  # ensure valid nodes
        G.add_edge(source, target, label=rel_type, type=rel_type)

# Save to GEXF (supports labels/hover)
os.makedirs("outputs", exist_ok=True)
nx.write_gexf(G, graph_path)

print(f"✅ Graph with {len(G.nodes)} nodes and {len(G.edges)} edges saved → {graph_path}")
print("💡 Open this file in Gephi or PyVis to hover and explore nodes/edges with labels.")

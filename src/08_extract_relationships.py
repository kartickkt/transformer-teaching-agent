import json
from pathlib import Path
from itertools import combinations
from openai import OpenAI
from dotenv import load_dotenv
import time
import re
import numpy as np

# ----------------------
# Load OpenAI key
# ----------------------
load_dotenv()
client = OpenAI()

# ----------------------
# Paths
# ----------------------
concepts_path = Path("outputs/05_final_concepts.json")
embeddings_path = Path("temp/06_final_embeddings.json")
similarity_graph_path = Path("temp/07_concept_graph.json")
enriched_graph_path = Path("outputs/08_concept_graph_enriched.json")
cache_dir = Path("temp/08_relationship_cache")
cache_dir.mkdir(exist_ok=True)

# ----------------------
# Load concepts
# ----------------------
with open(concepts_path, "r") as f:
    concepts = json.load(f)
print(f"Loaded {len(concepts)} concepts")

# ----------------------
# Load embeddings
# ----------------------
with open(embeddings_path, "r") as f:
    data = json.load(f)

# Convert all embeddings to float arrays
embeddings = {concept: np.array(vec, dtype=float) for concept, vec in data.items()}
print(f"Loaded {len(embeddings)} embeddings")

# ----------------------
# Load existing similarity graph
# ----------------------
with open(similarity_graph_path, "r") as f:
    similarity_graph = json.load(f)

similarity_edges = similarity_graph["edges"]
sim_pairs = [(e["source"], e["target"]) for e in similarity_edges]
print(f"Loaded {len(sim_pairs)} similarity edges")

# ----------------------
# Safe JSON parser
# ----------------------
def safe_json_loads(text):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    return []

# ----------------------
# Cosine similarity helper
# ----------------------
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ----------------------
# Generate candidate prerequisite pairs (threshold 0.35)
# ----------------------
similarity_threshold = 0.35
candidate_pairs = []

for i, c1 in enumerate(concepts):
    for j, c2 in enumerate(concepts):
        if i == j:
            continue
        sim = cosine_sim(embeddings[c1], embeddings[c2])
        if sim >= similarity_threshold:
            candidate_pairs.append((c1, c2))

print(f"Candidate prerequisite pairs: {len(candidate_pairs)}")

# ----------------------
# LLM query helper
# ----------------------
def get_relationships(batch_pairs, relation_types=["prerequisite"], cache_name=None):
    # Check cache first
    if cache_name:
        cache_file = cache_dir / f"{cache_name}.json"
        if cache_file.exists():
            with open(cache_file, "r") as f:
                return json.load(f)

    prompt = f"""
Given the following concept pairs: {batch_pairs}

For each pair, return their relationship as one of:
{', '.join([f'"{r}"' for r in relation_types])} or "none".

Return ONLY valid JSON array like:
[
  {{"source": "ConceptA", "target": "ConceptB", "relation": "prerequisite"}},
  ...
]

Only include pairs where relation is not "none".
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        text = response.choices[0].message.content
        rels = safe_json_loads(text)
        if cache_name:
            with open(cache_file, "w") as f:
                json.dump(rels, f, indent=2)
        return rels
    except Exception as e:
        print(f"⚠️ LLM request failed: {e}")
        return []

# ----------------------
# Step 1: Extract prerequisite edges
# ----------------------
batch_size = 5
prereq_edges = []

for i in range(0, len(candidate_pairs), batch_size):
    batch = candidate_pairs[i:i+batch_size]
    cache_name = f"prereq_batch_{i//batch_size}"
    print(f"Querying prerequisite batch {i//batch_size + 1} / {((len(candidate_pairs)-1)//batch_size)+1}")
    rels = get_relationships(batch, relation_types=["prerequisite"], cache_name=cache_name)
    prereq_edges.extend(rels)
    time.sleep(0.5)  # avoid rate limits

print(f"Collected {len(prereq_edges)} prerequisite edges")

# ----------------------
# Step 2: Enrich existing similarity edges
# ----------------------
enrichment_types = ["example_of", "related_topic", "application_of", "contrasts_with"]
sim_enriched_edges = []

for i in range(0, len(sim_pairs), batch_size):
    batch = sim_pairs[i:i+batch_size]
    cache_name = f"sim_enrich_batch_{i//batch_size}"
    print(f"Querying similarity batch {i//batch_size + 1} / {((len(sim_pairs)-1)//batch_size)+1}")
    rels = get_relationships(batch, relation_types=enrichment_types, cache_name=cache_name)
    # Add similarity weight from original edge
    for r in rels:
        weight = next((e["weight"] for e in similarity_edges
                       if (e["source"], e["target"]) == (r["source"], r["target"]) or
                          (e["source"], e["target"]) == (r["target"], r["source"])), None)
        if weight is not None:
            r["weight"] = weight
            sim_enriched_edges.append(r)
    time.sleep(0.5)

print(f"Collected {len(sim_enriched_edges)} enriched similarity edges")

# ----------------------
# Step 3: Combine nodes and edges
# ----------------------
nodes = [{"id": c, "label": c} for c in concepts]
edges = prereq_edges + sim_enriched_edges

enriched_graph = {
    "nodes": nodes,
    "edges": edges
}

# Save final enriched graph
with open(enriched_graph_path, "w") as f:
    json.dump(enriched_graph, f, indent=2)

print(f"✅ Enriched graph saved to {enriched_graph_path}")
print(f"Nodes: {len(nodes)}, Edges: {len(edges)}")

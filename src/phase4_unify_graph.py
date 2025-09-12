import json
import numpy as np
import networkx as nx
from openai import OpenAI
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
import os

load_dotenv()  # reads .env file
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


# ----------------- Settings -----------------
INPUT_FILE = "outputs/phase3_normalized.json"
UNIFIED_JSON = "outputs/phase4_unified.json"
UNIFIED_GRAPH = "outputs/knowledge_graph_unified.gexf"
SIMILARITY_THRESHOLD = 0.9

# ----------------- Load OpenAI client -----------------
client = OpenAI()  # make sure OPENAI_API_KEY is set in .env

# ----------------- Load extractions -----------------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    extractions = json.load(f)

# ----------------- Extract concept texts -----------------
concept_indices = [i for i, ext in enumerate(extractions) if ext["class"] == "concept"]
concept_texts = [extractions[i]["text"] for i in concept_indices]

print(f"💡 Computing embeddings for {len(concept_texts)} concepts...")

# ----------------- Compute embeddings -----------------
resp = client.embeddings.create(
    model="text-embedding-3-large",
    input=concept_texts
)
embeddings = [np.array(item.embedding) for item in resp.data]

# ----------------- Merge concepts -----------------
merged_map = {}  # old_idx -> new_idx

for i in range(len(concept_texts)):
    if i in merged_map:
        continue
    for j in range(i + 1, len(concept_texts)):
        if j in merged_map:
            continue
        sim = 1 - cosine(embeddings[i], embeddings[j])
        if sim >= SIMILARITY_THRESHOLD:
            merged_map[j] = i
            print(f"🔗 Merging '{concept_texts[j]}' into '{concept_texts[i]}' (similarity={sim:.3f})")

# ----------------- Apply merges -----------------
unified_extractions = []
for idx, ext in enumerate(extractions):
    if ext["class"] == "concept":
        # remap if merged
        new_idx = merged_map.get(idx, idx)
        if new_idx != idx:
            continue  # skip duplicates, merged into new_idx
        unified_extractions.append(ext)
    else:
        # relationships: remap subject/object if merged
        subj = ext["attributes"]["subject"]
        obj = ext["attributes"]["object"]

        # remap if subject/object were merged
        if isinstance(subj, str):
            if subj in [concept_texts[j] for j in merged_map.keys()]:
                j_idx = [concept_texts[j] for j in merged_map.keys()].index(subj)
                subj = concept_texts[merged_map[list(merged_map.keys())[j_idx]]]
        if isinstance(obj, str):
            if obj in [concept_texts[j] for j in merged_map.keys()]:
                j_idx = [concept_texts[j] for j in merged_map.keys()].index(obj)
                obj = concept_texts[merged_map[list(merged_map.keys())[j_idx]]]

        ext["attributes"]["subject"] = subj
        ext["attributes"]["object"] = obj
        unified_extractions.append(ext)

# ----------------- Save unified JSON -----------------
with open(UNIFIED_JSON, "w", encoding="utf-8") as f:
    json.dump(unified_extractions, f, indent=2)
print(f"💾 Saved unified extractions to {UNIFIED_JSON}")

# ----------------- Build networkx graph -----------------
def build_networkx_graph(extractions):
    G = nx.DiGraph()
    for ext in extractions:
        if ext["class"] == "concept":
            G.add_node(ext["text"], type="concept")
        elif ext["class"] == "relationship":
            subj = ext["attributes"]["subject"]
            obj = ext["attributes"]["object"]
            rel = ext["attributes"]["relation"]
            G.add_node(subj, type="concept")
            G.add_node(obj, type="concept")
            G.add_edge(subj, obj, predicate=rel)
    return G

G = build_networkx_graph(unified_extractions)
nx.write_gexf(G, UNIFIED_GRAPH)
print(f"✅ Built unified graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
print(f"💾 Saved graph to {UNIFIED_GRAPH}")

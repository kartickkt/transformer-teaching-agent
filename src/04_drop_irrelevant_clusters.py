# src/04_drop_irrelevant_clusters.py

import json
import numpy as np
from sklearn.cluster import KMeans
import openai
from pathlib import Path
from dotenv import load_dotenv
import os

# --- Load environment ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
openai.api_key = openai_api_key

# --- CONFIG ---
INPUT_FILE = "outputs/03_deduplicate_concepts_output.json"
CLUSTERS_FILE = "outputs/04_clusters.json"
OUTPUT_FILE = "outputs/04_cleaned_concepts.json"
N_CLUSTERS = 10
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# --- HELPER FUNCTIONS ---
def load_concepts(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        concepts = json.load(f)
    return concepts

def save_clusters(clusters, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)
    print(f"✅ Clusters saved → {file_path}")

def save_cleaned_concepts(concepts_to_keep, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(concepts_to_keep, f, indent=2, ensure_ascii=False)
    print(f"✅ Cleaned concepts saved → {file_path}")

def get_embeddings(concepts):
    embeddings = []
    for i, concept in enumerate(concepts):
        resp = openai.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=concept
        )
        embeddings.append(resp.data[0].embedding)
        if (i + 1) % 50 == 0 or i == len(concepts) - 1:
            print(f"Generated embeddings for {i+1}/{len(concepts)} concepts")
    return np.array(embeddings)

# --- MAIN SCRIPT ---
def main():
    concepts = load_concepts(INPUT_FILE)
    print(f"✅ Loaded {len(concepts)} deduplicated concepts")

    # Step 1: generate embeddings
    embeddings = get_embeddings(concepts)

    # Step 2: cluster embeddings
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Step 3: organize clusters
    clusters = {str(i): [] for i in range(N_CLUSTERS)}
    for concept, label in zip(concepts, cluster_labels):
        clusters[str(label)].append(concept)

    # Save all clusters for inspection
    save_clusters(clusters, CLUSTERS_FILE)

    print("\nClusters saved! Open the file to inspect all concepts in each cluster.")
    print("After inspection, enter the cluster numbers you want to KEEP.")

    keep_labels = input(f"\nEnter cluster numbers to keep (comma-separated, e.g., 0,2,4): ")
    keep_labels = [str(x.strip()) for x in keep_labels.split(",")]

    concepts_to_keep = []
    for label in keep_labels:
        concepts_to_keep.extend(clusters[label])

    print(f"\n✅ Retaining {len(concepts_to_keep)} concepts out of {len(concepts)}")

    # Step 4: save final cleaned concepts
    save_cleaned_concepts(concepts_to_keep, OUTPUT_FILE)

if __name__ == "__main__":
    main()

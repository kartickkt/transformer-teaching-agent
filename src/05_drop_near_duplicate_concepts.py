import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------------------
# Setup
# ---------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_FILE = "outputs/04_cleaned_concepts.json"
OUTPUT_FILE = "outputs/05_final_concepts.json"
EMBEDDINGS_FILE = "temp/05_embeddings.json"
MAPPING_FILE = "temp/05_near_duplicate_mapping.json"
BORDERLINE_FILE = "temp/05_borderline_pairs.json"

THRESHOLD = 0.80            # Deduplication threshold
BORDERLINE_RANGE = (0.78, 0.80)  # Borderline similarity range

# Ensure directories exist
Path("temp").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

# ---------------------------
# Helpers
# ---------------------------
def load_concepts(input_file):
    """Load concepts (expects list of strings)."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(c, str) for c in data):
        raise ValueError("Concepts file must contain a list of strings")
    return data


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_embeddings(concepts):
    """Fetch or load embeddings for concepts."""
    if Path(EMBEDDINGS_FILE).exists():
        with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    embeddings = []
    for concept in concepts:
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=concept
        )
        embeddings.append(resp.data[0].embedding)

    save_json(embeddings, EMBEDDINGS_FILE)
    return embeddings


def drop_near_duplicates(concepts, threshold=THRESHOLD):
    """
    Drop near duplicates using cosine similarity.

    Returns:
        kept: list of unique concepts
        mapping: dict of concept -> duplicates
        removed: set of removed concepts
        borderline_pairs: list of borderline similarity pairs
    """
    embeddings = get_embeddings(concepts)
    kept = []
    mapping = {}
    removed = set()
    borderline_pairs = []

    for i, concept in enumerate(concepts):
        if concept in removed:
            continue

        kept.append(concept)
        mapping[concept] = []

        if i + 1 >= len(concepts):
            continue

        current_emb = np.array(embeddings[i]).reshape(1, -1)
        later_embs = np.array(embeddings[i + 1:])

        if later_embs.shape[0] == 0:
            continue

        sims = cosine_similarity(current_emb, later_embs)[0]

        for j, sim in enumerate(sims, start=i + 1):
            if sim >= threshold:
                mapping[concept].append(concepts[j])
                removed.add(concepts[j])
                mapping[concept] = sorted(set(mapping[concept]))
            elif BORDERLINE_RANGE[0] <= sim < BORDERLINE_RANGE[1]:
                borderline_pairs.append({
                    "concept_1": concept,
                    "concept_2": concepts[j],
                    "similarity": round(sim, 2)
                })
                print(f"[Borderline] {concept} ↔ {concepts[j]} (sim={sim:.2f})")

    save_json(mapping, MAPPING_FILE)
    save_json(borderline_pairs, BORDERLINE_FILE)
    return kept, mapping, removed, borderline_pairs


# ---------------------------
# Main
# ---------------------------
def main():
    concepts = load_concepts(INPUT_FILE)
    unique_concepts, mapping, removed, borderline_pairs = drop_near_duplicates(concepts, THRESHOLD)
    save_json(unique_concepts, OUTPUT_FILE)

    print(f"\n✅ Cleaned concepts saved to {OUTPUT_FILE}")
    print(f"✅ Embeddings cached in {EMBEDDINGS_FILE}")
    print(f"✅ Mapping saved to {MAPPING_FILE}")
    print(f"✅ Borderline pairs saved to {BORDERLINE_FILE}")

    # --- Summary Report ---
    print("\n=== SUMMARY REPORT ===")
    print(f"Initial concepts: {len(concepts)}")
    print(f"Final concepts:   {len(unique_concepts)}")
    print(f"Removed concepts: {len(removed)}")
    print(f"Deduplication threshold: {THRESHOLD}")
    print(f"Borderline pairs: {len(borderline_pairs)}")
    print("=======================")


if __name__ == "__main__":
    main()

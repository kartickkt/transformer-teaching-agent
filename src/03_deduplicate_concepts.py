# src/03_deduplicate_concepts.py
import json
import os
from collections import defaultdict

# ---------------- Paths ----------------
INPUT_PATH = "outputs/02_initial_concept_extract_output.json"
OUTPUT_PATH = "outputs/03_deduplicate_concepts_output.json"
MAPPING_PATH = "temp/03_deduplicate_mapping.json"  # just for inspection

# ---------------- Helpers ----------------
def normalize(text: str) -> str:
    """Normalize text for duplicate detection."""
    return (
        text.lower()
        .replace("the ", "")
        .replace("a ", "")
        .replace("an ", "")
        .replace("(", "")
        .replace(")", "")
        .replace("-", " ")
        .replace("_", " ")
        .replace(",", "")
        .strip()
    )

def deduplicate(concepts):
    seen = {}
    kept = []
    dropped = []
    mapping = defaultdict(list)

    for concept in concepts:
        norm = normalize(concept)
        if norm not in seen:
            seen[norm] = concept  # keep original form
            kept.append(concept)
        else:
            dropped.append(concept)
            mapping[seen[norm]].append(concept)

    return kept, dropped, mapping

# ---------------- Main ----------------
def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        concepts = json.load(f)

    print(f"✅ Loaded {len(concepts)} raw concepts")

    kept, dropped, mapping = deduplicate(concepts)

    print(f"✅ Deduplicated concepts: kept {len(kept)}, dropped {len(dropped)}")

    # Save deduplicated output
    os.makedirs("outputs", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)

    # Save mapping of kept → dropped variants
    os.makedirs("temp", exist_ok=True)
    with open(MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"✅ Deduplicated file saved → {OUTPUT_PATH}")
    print(f"✅ Duplicate mapping saved → {MAPPING_PATH}")

    if dropped:
        print("\nDropped examples:")
        for d in dropped[:20]:
            print(" -", d)

if __name__ == "__main__":
    main()

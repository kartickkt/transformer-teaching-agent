# scripts/phase2_relationships.py

import json
import re
import os
from pathlib import Path
import sys
from tqdm import tqdm

# ---------------- Path setup for Colab / local ----------------
repo_root = Path(__file__).resolve().parents[1]  # go up one level from scripts/
sys.path.append(str(repo_root))

from src.utils import load_json, save_json
from src.llm_utils import call_llm

# ---------------- Files & settings ----------------
INPUT_FILE = "outputs/phase2_concepts.json"
OUTPUT_FILE = "outputs/phase2_relationships.json"
BATCH_SIZE = 10  # smaller batches improve JSON compliance


# ---------------- JSON repair helper ----------------
def safe_json_parse(response: str):
    try:
        return json.loads(response)
    except Exception:
        # Try to extract JSON array inside text
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return []
        return []


# ---------------- Prompt ----------------
def make_prompt(concepts):
    return (
        "Role: You are a teacher on Transformer architecture in deep learning.\n"
        "Task: From the following concepts, extract relationships and dependencies between them, "
        "so that you can teach a student in a systematic way.\n"
        "Focus on: hierarchical (part_of), dependency (depends_on), usage (used_for), and analogical (similar_to) relationships.\n"
        "Special instruction: If a concept appears in multiple places (e.g., Multi-head attention is part of both "
        "encoder and decoder layers), list all applicable relationships.\n"
        "Return ONLY a valid JSON array in this exact format:\n"
        "[\n"
        "  {\"subject\": \"ConceptA\", \"relation\": \"depends_on\", \"object\": \"ConceptB\"},\n"
        "  {\"subject\": \"ConceptC\", \"relation\": \"part_of\", \"object\": \"ConceptD\"}\n"
        "]\n\n"
        "Guidelines:\n"
        "- Use ONLY the provided concepts.\n"
        "- Relations must be concise and meaningful.\n"
        "- No self-relations (ConceptA → ConceptA).\n"
        "- Avoid duplicates or reversed duplicates unless direction matters.\n\n"
        f"Concept list:\n{json.dumps(concepts, indent=2)}\n"
    )


# ---------------- Main logic ----------------
def extract_relationships(concepts):
    relationships = []

    for i in tqdm(range(0, len(concepts), BATCH_SIZE)):
        batch = concepts[i : i + BATCH_SIZE]
        prompt = make_prompt(batch)
        response = call_llm(prompt)

        rels = safe_json_parse(response)
        if isinstance(rels, list):
            relationships.extend(rels)
        else:
            print("⚠️ Could not parse relationships for this batch.")

    return relationships


def main():
    print("📂 Loading concepts...")
    concepts = load_json(INPUT_FILE)

    if isinstance(concepts, dict) and "concepts" in concepts:
        concepts = concepts["concepts"]

    print(f"✅ Loaded {len(concepts)} concepts")

    print("🔗 Extracting relationships...")
    relationships = extract_relationships(concepts)

    print(f"✅ Extracted {len(relationships)} relationships")

    save_json(relationships, OUTPUT_FILE)
    print(f"💾 Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

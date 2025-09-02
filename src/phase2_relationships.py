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
    """
    Few-shot prompt for extracting concept relationships for a teaching agent.
    """
    examples = [
        {
            "concepts": ["Transformer encoder layer", "Multi-head attention", "Position-wise feed-forward"],
            "relationships": [
                {"subject": "Multi-head attention", "relation": "part_of", "object": "Transformer encoder layer"},
                {"subject": "Position-wise feed-forward", "relation": "part_of", "object": "Transformer encoder layer"}
            ]
        },
        {
            "concepts": ["Self-attention", "Query", "Key", "Value"],
            "relationships": [
                {"subject": "Query", "relation": "used_for", "object": "Self-attention"},
                {"subject": "Key", "relation": "used_for", "object": "Self-attention"},
                {"subject": "Value", "relation": "used_for", "object": "Self-attention"}
            ]
        }
    ]

    # Convert few-shot examples to text
    example_text = ""
    for ex in examples:
        example_text += "Concepts: " + ", ".join(ex["concepts"]) + "\n"
        example_text += "Relationships: " + str(ex["relationships"]) + "\n\n"

    # Prompt
    prompt = f"""
Role: You are a teacher in deep learning and transformer architectures.
Task: From the following list of concepts, extract relationships that help a student understand
the material systematically.

Focus on: hierarchical (part_of), dependency (depends_on), usage (used_for), and analogical (similar_to) relationships.

Output format: return ONLY a JSON array with objects like:
[{{"subject": "ConceptA", "relation": "depends_on", "object": "ConceptB"}}]

Examples:
{example_text}

Concepts to analyze:
{json.dumps(concepts, indent=2)}

Relationships:
"""
    return prompt



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

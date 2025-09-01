import os
from pathlib import Path
from tqdm import tqdm

from utils import load_json, save_json
from llm_utils import call_llm

# ---------------- Paths ----------------
REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = REPO_ROOT / "outputs" / "attention_chunks.json"
OUTPUT_FILE = REPO_ROOT / "outputs" / "phase2_concepts.json"

# ---------------- LLM Settings ----------------
LLM_TEMPERATURE = 0.0  # deterministic extraction

# ---------------- Concept Extraction Prompt ----------------
def build_concept_prompt(chunk_text: str) -> str:
    """
    Builds the prompt for the LLM to extract key concepts from a text chunk.
    """
    return (
        "Extract only the important technical concepts related to deep learning, "
        "transformers, large language models, mathematics from the following text. "
        "Do not include names, email addresses, citations or unrelated words. "
        "Return them as a JSON array of short strings, without any explanation.\n\n"
        f"Text:\n{chunk_text}\n\nConcepts:"
    )

# ---------------- Phase 2 Pipeline ----------------
def extract_concepts_from_chunks(chunks: list) -> list:
    """
    Takes a list of text chunks and extracts key concepts from each using the LLM.
    Returns a list of dictionaries with 'chunk_id', 'chunk', and 'concepts'.
    """
    results = []
    for entry in tqdm(chunks, desc="Extracting concepts"):
        chunk_id = entry.get("chunk_id")
        chunk_text = entry.get("chunk")

        prompt = build_concept_prompt(chunk_text)
        response = call_llm(prompt, temperature=LLM_TEMPERATURE, do_sample=False)

        # Attempt to parse JSON from LLM response
        try:
            import json
            concepts = json.loads(response)
            if not isinstance(concepts, list):
                raise ValueError
        except Exception:
            # fallback: split by commas if JSON fails
            concepts = [c.strip() for c in response.strip("[]").split(",") if c.strip()]

        results.append({
            "chunk_id": chunk_id,
            "chunk": chunk_text,
            "concepts": concepts
        })
    return results

# ---------------- Main ----------------
def main():
    # Load Phase 1 chunks
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Phase 1 output not found at {INPUT_FILE}")

    print(f"Loading Phase 1 chunks from {INPUT_FILE}")
    chunks_data = load_json(INPUT_FILE)

    if not isinstance(chunks_data, list):
        raise ValueError("Phase 1 JSON must be a list of chunks")

    # Extract concepts
    phase2_output = extract_concepts_from_chunks(chunks_data)

    # Save Phase 2 output
    save_json(phase2_output, OUTPUT_FILE)
    print(f"Phase 2 concepts saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

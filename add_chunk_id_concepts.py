import json

INPUT_FILE = "outputs/phase2_concepts.json"   # your good output file
OUTPUT_FILE = "outputs/phase2_concepts_with_ids.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Add chunk_id based on index
for i, item in enumerate(data):
    item["chunk_id"] = i

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"✅ Added chunk_id to {len(data)} chunks. Saved to {OUTPUT_FILE}")

import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
client = OpenAI()

# Paths
concepts_path = Path("outputs/05_final_concepts.json")
embeddings_path = Path("temp/06_final_embeddings.json")

# Load final concepts
with open(concepts_path, "r") as f:
    concepts = json.load(f)

print(f"Loaded {len(concepts)} final concepts")

def get_embedding(text: str, model="text-embedding-3-small"):
    """Fetch embedding for a given text using OpenAI."""
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

# Dictionary to store embeddings
full_embeddings = {}

for i, concept in enumerate(concepts, 1):
    embedding = get_embedding(concept)
    full_embeddings[concept] = embedding
    if i % 25 == 0 or i == len(concepts):
        print(f"Processed {i}/{len(concepts)} concepts")

# Save embeddings as pretty JSON
with open(embeddings_path, "w") as f:
    json.dump(full_embeddings, f, indent=2)

print(f"✅ Saved embeddings to {embeddings_path}")

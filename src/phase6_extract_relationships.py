import os
import json
import time
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load concepts
with open("outputs/final_concepts.json", "r", encoding="utf-8") as f:
    concepts = json.load(f)

final_relationships = []

def clean_json_response(raw_content: str) -> str:
    """Remove markdown fences and extra text around JSON."""
    raw_content = raw_content.strip()
    if raw_content.startswith("```"):
        # Remove triple backticks and optional 'json'
        raw_content = raw_content.strip("`")
        raw_content = raw_content.replace("json", "", 1).strip()
    return raw_content

def extract_relationships(concept, retries=3, delay=2):
    """Extract relationships for a single concept with retries for JSON errors."""
    prompt = f"""
    List relationships for the concept '{concept}' in strict JSON array format.
    Each item should be of the form:
    {{
      "source": "{concept}",
      "target": "<related_concept>",
      "type": "<relation_type>"
    }}
    Respond with ONLY valid JSON.
    """

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800
            )
            raw_content = response.choices[0].message.content
            print(f"\n📝 Raw response for '{concept}':\n{raw_content}\n")

            cleaned = clean_json_response(raw_content)
            relationships = json.loads(cleaned)
            return relationships

        except json.JSONDecodeError:
            print(f"⚠️ JSON decode error for '{concept}', attempt {attempt+1}/{retries}")
            time.sleep(delay)
        except Exception as e:
            print(f"⚠️ OpenAI/API error for '{concept}': {e}, attempt {attempt+1}/{retries}")
            time.sleep(delay)

    print(f"❌ Failed to extract relationships for '{concept}' after {retries} retries.")
    return []

# Process all concepts (slice [:5] if testing)
for concept in tqdm(concepts, desc="Extracting"):
    relationships = extract_relationships(concept)
    final_relationships.extend(relationships)

# Save results
os.makedirs("outputs", exist_ok=True)
with open("outputs/final_relationships.json", "w", encoding="utf-8") as f:
    json.dump(final_relationships, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(final_relationships)} relationships to outputs/final_relationships.json")

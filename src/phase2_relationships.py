import json
import os
from llama_index.core import Document, KnowledgeGraphIndex, StorageContext
from llama_index.core.llms import MistralAI
from llama_index.core.graph_stores import SimpleGraphStore

# --- Load the JSON data from the file ---
# Define the path to the JSON file
file_path = os.path.join("outputs", "phase2_concepts.json")
with open(file_path, 'r') as f:
    concepts_json = json.load(f)

# Read the data from the file
try:
    with open(file_path, 'r') as f:
        concepts_json = json.load(f)
except FileNotFoundError:
    print(f"Error: The file {file_path} was not found.")
    exit()

# --- Setup ---
# Define the LLM
# Ensure you have your MISTRAL_API_KEY set up as an environment variable
llm = MistralAI(model="mistralai/Mistral-7B-Instruct-v0.3")

# Define your documents from the chunks
documents = [
    Document(text=d["chunk"], metadata={"chunk_id": d["chunk_id"]})
    for d in concepts_json
]

# --- Build the Graph ---
# Define a simple graph store
graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# Define the relationship extraction schema
entity_types = ["Concept"]
relationship_types = ["IS_A", "REPLACES", "BENEFITS_FROM", "INVOLVES", "ACHIEVES"]

# Create the Knowledge Graph Index
print("Building the Knowledge Graph... This may take some time.")
index = KnowledgeGraphIndex.from_documents(
    documents,
    llm=llm,
    storage_context=storage_context,
    kg_extractors=KnowledgeGraphIndex.get_kg_extractors(
        kg_extractors_list=["simple", "schema_enforced"],
        llm=llm,
        kg_rel_types=relationship_types,
    ),
)
print("Knowledge Graph built successfully! 🎉")

# --- Save the Output ---
persist_dir = "./attention_kg_data"

if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)

print(f"Persisting the knowledge graph to {persist_dir}...")
index.storage_context.persist(persist_dir=persist_dir)
print("Done! The graph has been saved.")
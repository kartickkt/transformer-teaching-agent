from llama_index.core import Document, KnowledgeGraphIndex, StorageContext
from llama_index.core.llms import MistralAI
from llama_index.core.graph_stores import SimpleGraphStore
import os

# --- Setup ---
# Define the LLM
# This setup assumes you have the Mistral API key configured.
# You might need to set it up like: os.environ['MISTRAL_API_KEY'] = 'YOUR_API_KEY'
llm = MistralAI(model="mistralai/Mistral-7B-Instruct-v0.3")

# Define your documents from the chunks
# This part remains the same
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
# 1. Define the directory to save the files
persist_dir = "outputs/attention_kg_data"

# Ensure the directory exists
if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)

# 2. Persist the index to disk
print(f"Persisting the knowledge graph to {persist_dir}...")
index.storage_context.persist(persist_dir=persist_dir)
print("Done! The graph has been saved to your local file system.")

# --- Example of Loading it back ---
# To demonstrate, here's how you would load the saved graph later
# print("Now loading the saved graph from disk...")
# loaded_storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
# loaded_index = load_index_from_storage(loaded_storage_context)
# print("Graph loaded successfully!")
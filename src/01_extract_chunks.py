# ================== Setup ==================
import os
import json
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# ----------------- Hugging Face / OpenAI tokens -----------------
# Make sure to export your tokens in env variables before running
# os.environ["OPENAI_API_KEY"] = "<YOUR_OPENAI_KEY>"
# Optional if using HF models: os.environ["HF_TOKEN"] = "<YOUR_HF_TOKEN>"

# ================== Phase 1 Agent ==================
class Phase1ChunkingAgent:
    def __init__(self, chunk_size=400, chunk_overlap=50, use_llm_filter=True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_llm_filter = use_llm_filter
        self.llm = None
        if self.use_llm_filter:
            self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    # ----- PDF extraction -----
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    # ----- Chunking -----
    def chunk_text(self, text: str):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " "]
        )
        return splitter.split_text(text)

    # ----- Optional LLM filtering -----
    def filter_chunks(self, chunks):
        if not self.use_llm_filter:
            return chunks

        prompt_template = PromptTemplate(
            input_variables=["chunk"],
            template="""
You are given a text chunk. 
Return 'VALID' if the chunk contains meaningful content suitable for teaching. 
Return 'INVALID' if it is empty, only headings, numbers, or low-content.

Chunk:
{chunk}
"""
        )

        filtered = []
        for chunk in chunks:
            response = self.llm(prompt_template.format(chunk=chunk))
            if "VALID" in response.upper():
                filtered.append(chunk)
        return filtered

    # ----- Save chunks -----
    def save_chunks(self, chunks, save_path="attention_chunks.json"):
        with open(save_path, "w") as f:
            json.dump(chunks, f, indent=2)
        print(f"✅ Saved {len(chunks)} chunks to {save_path}")

# ================== Example Usage ==================
agent = Phase1ChunkingAgent(chunk_size=400, chunk_overlap=50, use_llm_filter=False)

pdf_path = "data/attention.pdf"
text = agent.extract_text_from_pdf(pdf_path)
chunks = agent.chunk_text(text)
# Optional LLM filtering
chunks = agent.filter_chunks(chunks)
agent.save_chunks(chunks, save_path="outputs/attention_chunks.json")

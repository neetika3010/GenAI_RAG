## GenAI_RAG – Retrieval-Augmented Generation Pipeline
## Objective:
Build a Retrieval-Augmented Generation (RAG) system that can ingest documents, store them in a vector database, and answer natural language queries grounded in the retrieved context.
Designed as a modular, production-ready pipeline to showcase Generative AI engineering and LLM application skills.

## Key Highlights
- End-to-End RAG Pipeline : from document ingestion to final answer generation.
- Multi-Source Document Loader: supports PDF, TXT, Markdown, CSV.
- Custom Chunking Strategy:optimized for LLM context windows.
- Embeddings Layer: OpenAI or HuggingFace models.
- Vector Store :FAISS for local dev, easily swappable for Weaviate/Pinecone.
- LLM Integration :tested with Mistral-7B.
- Deployment Ready :CLI tools + REST API (FastAPI) + .env configuration.

##Tech Stack
  | Layer               | Technology                                           |
| ------------------- | ---------------------------------------------------- |
| **Language Models** |  OpenAI GPT-4o-mini, Mistral-7B                      |
| **Embeddings**      | Sentence-Transformers |
| **Vector DB**       | FAISS (local), Pinecone/Weaviate (optional)          |
| **Frameworks**      | LangChain, FastAPI                           |
| **Utilities**       | Python-dotenv                             |
|                 |

## Workflow
## Ingest
- Load documents from data/raw/
- Chunk text (size + overlap tuned for model context)

## Generate embeddings
- Store in FAISS index

## Retrieve
- User query → embed → similarity search (top-k chunks)
- Return context + metadata

## Generate
- Prompt LLM with retrieved context
- Generate grounded answer with source citations

Documents --> Loader --> Chunker --> Embeddings --> Vector Store (FAISS)
                                                        ↑
                                                        |
                                                   Similarity Search
                                                        |
User Query --> Embeddings --> Retriever --> Prompt --> LLM --> Answer + Sources


## Future Improvements
- Add Reranking :Integrate advanced rerankers like BGE-Reranker or Cohere Rerank to improve retrieval quality by reordering top-k search results based on semantic relevance.
- Deploy via Docker & Cloud – Containerize the pipeline with Docker and deploy to AWS/GCP/Azure for scalable, production-ready RAG services.
- Integrate RAG Evaluation (RAGAS) – Incorporate RAGAS metrics to evaluate retrieval quality, faithfulness, and answer relevance in an automated manner.

## Quickstart

- Install Requirements
  pip install -r requirements.txt

- Configure Environment
  Create .env:
  
PROVIDER=openai
OPENAI_API_KEY=
EMBEDDINGS_MODEL=text-embedding-3-small
GEN_MODEL=gpt-4o-mini
VECTOR_DB=faiss
FAISS_INDEX_DIR=./index
CHUNK_SIZE=800
CHUNK_OVERLAP=120
TOP_K=4

- Add Documents
  Place your files in:data/raw/

- Build the Index
  python -m src.cli ingest --input data/raw


- Ask a Question (CLI)
  python -m src.cli ask "What are the key points in the document?"








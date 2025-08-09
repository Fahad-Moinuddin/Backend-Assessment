# **RAG Chatbot Backend**

A minimal viable **Retrieval-Augmented Generation (RAG)** backend built with **FastAPI**, integrating **OpenAI** (or AWS Bedrock) for LLM queries, **AWS S3** for vector storage, and **cosine similarity search** for context retrieval.

This project implements:

* REST API endpoints for chat, document upload, and vector search.
* Vector storage in S3 (JSON format).
* Document chunking for better retrieval.
* Context-aware LLM responses.
* Background upload processing.
* Basic unit tests.

---

## **Features**

* **POST `/chat`** – Ask a question and get an LLM answer with retrieved context.
* **POST `/documents/upload`** – Upload a document, chunk it, embed it, and store it in S3.
* **GET `/documents/search`** – Search stored vectors for top-N relevant chunks.
* **Document chunking** – Splits documents into overlapping segments for better search results.
* **Background tasks** – Uploads and embeddings run in the background for faster API response.
* **Service separation** – LLM, vector store, and utilities are modular.

---

## **Tech Stack**

* **Python 3.10+**
* **FastAPI** (REST API)
* **OpenAI API** or AWS Bedrock (LLM + embeddings)
* **AWS S3** (vector storage)
* **NumPy** (vector math)
* **Custom cosine similarity** (no heavy dependencies)

---

## **Project Structure**

```
backend/
  ├── main.py                  # FastAPI app
  ├── services/
  │     ├── llm_service.py     # OpenAI/AWS LLM integration
  │     ├── vector_store.py    # S3 storage & retrieval
  ├── utils/
  │     ├── similarity.py      # Cosine similarity functions
  │     ├── embeddings.py      # Embedding generation
  │     ├── chunking.py        # Document chunking
  ├── tests/
  │     └── test_chat.py       # Example unit test
  ├── .env.example
  └── README.md
```

---

## **Setup Instructions**

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/rag-chatbot-backend.git
cd rag-chatbot-backend
```

### 2️⃣ Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate     # Linux/Mac
.venv\Scripts\activate        # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

If no `requirements.txt` exists, install manually:

```bash
pip install fastapi uvicorn boto3 numpy python-dotenv
pip install openai  # If using OpenAI embeddings
```

### 4️⃣ Configure environment variables

Create a `.env` file from `.env.example`:

```env
OPENAI_API_KEY=your-openai-key
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
AWS_S3_BUCKET=your-s3-bucket
```

---

## **Running the Backend**

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

Open API docs in browser:

```
http://127.0.0.1:8000/docs
```

---

## **API Endpoints**

### **1. Chat**

**POST** `/chat`
Send a question and get a context-aware answer.

```json
{
  "message": "What is machine learning?"
}
```

Response:

```json
{
  "answer": "Machine learning is ...",
  "context_used": ["Doc snippet 1", "Doc snippet 2"]
}
```

---

### **2. Upload Document**

**POST** `/documents/upload`
Upload a text file (chunks are created automatically, embeddings are generated in background).

```
multipart/form-data:
  file: your_document.txt
```

Response:

```json
{
  "status": "upload scheduled",
  "filename": "your_document.txt",
  "chunks": 5
}
```

---

### **3. Search Documents**

**GET** `/documents/search?query=your+search&top_n=3`
Find the most relevant chunks to a query.

```json
{
  "results": ["Snippet 1", "Snippet 2", "Snippet 3"],
  "count": 3
}
```

---

## **Testing**

Run unit tests:

```bash
pytest tests/
```

---

## **Notes**

* For the assessment, **embeddings can be faked** using `np.random.rand(768)` if Bedrock/OpenAI embeddings are not set up.
* To switch to real embeddings, update `utils/embeddings.py` to use AWS Bedrock or OpenAI API.
* All vector storage is in JSON format inside the S3 bucket under the `vectors/` prefix.

---

## **Bonus Features Implemented**

* ✅ Background processing for uploads.
* ✅ Document chunking.
* ✅ Modular code structure.
* ✅ `.env.example` included.

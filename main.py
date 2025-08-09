from fastapi import FastAPI, UploadFile, Form, HTTPException, BackgroundTasks
from services.llm_service import ask_llm
from services.vector_store import upload_document, search_vectors
from utils.chunking import chunk_text
from utils.embeddings import generate_embedding

app = FastAPI()

@app.post("/chat")
async def chat(message: str = Form(...)):
    try:
        # Generate embedding for the user's message
        query_embedding = generate_embedding(message)

        #Search S3 vector store for relevant docs
        context_chunks = search_vectors(query_embedding)

        #Build context string for LLM
        context_str = "\n".join(context_chunks) if context_chunks else "No relevant documents found."

        #Send Question + context to LLM
        prompt = f"Use the following context to answer the question:\n\n{context_str}\n\nQuestion: {message}"
        answer = ask_llm(prompt)

        return {"answer": answer, "context_used": context_chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/upload")
async def upload(file: UploadFile, background_tasks: BackgroundTasks):
    try:
        # Read file contents
        text = (await file.read()).decode()

        # Split into chunks for finer-grained retrieval
        chunks = chunk_text(text, chunk_size=500, overlap=50)

        # Schedule background embedding & upload
        background_tasks.add_task(process_and_upload, file.filename, chunks)

        return {"status": "upload scheduled", "filename": file.filename, "chunks": len(chunks)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/search")
async def search(query: str, top_n: int = 3):
    try:
        # Generate query embedding
        embedding = generate_embedding(query)

        # Get top_n results
        results = search_vectors(embedding, top_n=top_n)

        return {"results": results, "count": len(results)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Helper function for background processing
def process_and_upload(filename: str, chunks: list[str]):
    embeddings = [generate_embedding(chunk) for chunk in chunks]
    upload_document(filename, chunks, embeddings)


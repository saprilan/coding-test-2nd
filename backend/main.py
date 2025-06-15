from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from models.schemas import ChatRequest, ChatResponse, DocumentsResponse, UploadResponse
from services.pdf_processor import PDFProcessor
from services.vector_store import VectorStoreService
from services.rag_pipeline import RAGPipeline
from config import settings
import logging
import time
import os
import shutil

# Configure logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG-based Financial Statement Q&A System",
    description="AI-powered Q&A system for financial documents using RAG",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
pdf_processor: PDFProcessor = None
vector_store: VectorStoreService = None
rag_pipeline: RAGPipeline = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global pdf_processor, vector_store, rag_pipeline

    logger.info("Starting RAG Q&A System...")

    pdf_processor = PDFProcessor()
    vector_store = VectorStoreService()
    rag_pipeline = RAGPipeline()

    # Create upload path directory if not exists
    os.makedirs(settings.pdf_upload_path, exist_ok=True)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "RAG-based Financial Statement Q&A System is running"}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF file"""
    start_time = time.time()

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_path = os.path.join(settings.pdf_upload_path, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info("File saved: %s", file_path)

        # Process PDF
        documents = pdf_processor.process_pdf(file_path)

        print("after process document")

        # Add documents to vector store
        print(vector_store)
        vector_store.add_documents(documents)

        duration = time.time() - start_time
        return UploadResponse(
            filename=file.filename,
            total_chunks=len(documents),
            duration=round(duration, 2),
            status="success"
        )

    except Exception as e:
        logger.error("Failed to process PDF: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to process PDF")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat request and return AI response"""
    try:
        result = rag_pipeline.generate_answer(request.question, request.chat_history)
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            document_count=result["document_count"]
        )
    except Exception as e:
        logger.error("Chat request failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to generate answer")


@app.get("/api/documents", response_model=DocumentsResponse)
async def get_documents():
    """Get list of processed documents"""
    try:
        files = os.listdir(settings.pdf_upload_path)
        pdf_files = [f for f in files if f.lower().endswith(".pdf")]
        return DocumentsResponse(files=pdf_files, count=len(pdf_files))
    except Exception as e:
        logger.error("Failed to list documents: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to list documents")


@app.get("/api/chunks")
async def get_chunks():
    """Get document chunks (optional endpoint)"""
    try:
        count = vector_store.get_document_count()
        return {"chunk_count": count}
    except Exception as e:
        logger.error("Failed to count chunks: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to count chunks")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port, reload=settings.debug)

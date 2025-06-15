import os
from typing import List, Dict, Any
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import settings
import logging

logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(self):
        # Initialize text splitter with chunk size and overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "]
        )

    def extract_text_from_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF and return page-wise content"""
        pages_content = []

        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages_content.append({
                            "page_number": i + 1,
                            "text": text
                        })

            logger.info("Extracted %d pages from PDF: %s", len(pages_content), file_path)
        except Exception as e:
            logger.error("Failed to extract text from PDF %s: %s", file_path, str(e))

        return pages_content

    def split_into_chunks(self, pages_content: List[Dict[str, Any]]) -> List[Document]:
        """Split page content into chunks"""
        documents = []

        for page in pages_content:
            chunks = self.text_splitter.split_text(page["text"])
            for idx, chunk in enumerate(chunks):
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": os.path.basename(page.get("file_path", "unknown.pdf")),
                        "page_number": page["page_number"],
                        "chunk_index": idx
                    }
                ))

        logger.info("Split PDF into %d text chunks", len(documents))
        return documents

    def process_pdf(self, file_path: str) -> List[Document]:
        """Process PDF file and return list of Document objects"""
        pages = self.extract_text_from_pdf(file_path)

        # Add file_path metadata for use in chunk metadata
        for page in pages:
            page["file_path"] = file_path

        documents = self.split_into_chunks(pages)
        return documents

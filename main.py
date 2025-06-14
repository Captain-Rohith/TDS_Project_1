import base64
import io
import json
import numpy as np
import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from PIL import Image
import uvicorn
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="RAG Assistant", description="RAG-based Q&A system for student queries")

# Configure Gemini API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # base64 encoded image

class LinkResponse(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkResponse] = []

class RAGSystem:
    def __init__(self, embeddings_file: str = "markdown_embeddings.npz"):
        self.embeddings_file = embeddings_file
        self.embeddings = None
        self.metadata = None
        self.load_embeddings()
    
    def load_embeddings(self):
        """Load embeddings and metadata from npz file."""
        try:
            if not os.path.exists(self.embeddings_file):
                raise FileNotFoundError(f"Embeddings file {self.embeddings_file} not found")
            
            data = np.load(self.embeddings_file, allow_pickle=True)
            self.embeddings = data['embeddings']
            self.metadata = data['metadata']
            logger.info(f"Loaded {len(self.embeddings)} embeddings from {self.embeddings_file}")
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for the query."""
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query"
            )
            return np.array(result['embedding'])
        except Exception as e:
            logger.error(f"Error getting query embedding: {e}")
            raise
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def search_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find most similar chunks to the query."""
        query_embedding = self.get_query_embedding(query)
        
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            similarities.append({
                'index': i,
                'similarity': similarity,
                'file': str(self.metadata[i]['file']),
                'chunk_id': int(self.metadata[i]['chunk_id']),
                'text': str(self.metadata[i]['text'])
            })
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def extract_links_from_text(self, text: str) -> List[Dict[str, str]]:
        """Extract URLs and their context from text."""
        links = []
        
        # Pattern to match markdown links [text](url)
        markdown_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        matches = re.findall(markdown_pattern, text)
        for text_part, url in matches:
            links.append({"url": url, "text": text_part})
        
        # Pattern to match plain URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`[\]]+'
        urls = re.findall(url_pattern, text)
        for url in urls:
            # Get some context around the URL
            url_index = text.find(url)
            start = max(0, url_index - 50)
            end = min(len(text), url_index + len(url) + 50)
            context = text[start:end].strip()
            
            # Clean up context
            context = re.sub(r'\s+', ' ', context)
            if len(context) > 100:
                context = context[:97] + "..."
            
            links.append({"url": url, "text": context})
        
        return links
    
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]], image_data: Optional[str] = None) -> str:
        """Generate answer using Gemini with retrieved context."""
        try:
            # Prepare context
            context = "\n\n".join([
                f"From {chunk['file']} (similarity: {chunk['similarity']:.3f}):\n{chunk['text']}"
                for chunk in context_chunks
            ])
            
            # Create prompt
            prompt = f"""You are a helpful teaching assistant answering student questions based on course materials.

Context from course materials:
{context}

Student Question: {query}

Instructions:
1. Answer the question based primarily on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Be specific in your answers and give link to images where the answer is present when possible.
4. Keep the answer concise but complete
5. If you see conflicting information, point it out


Answer:"""

            # Handle image if provided
            if image_data:
                try:
                    # Decode base64 image
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Use Gemini Vision model
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content([prompt, image])
                    return response.text
                except Exception as e:
                    logger.error(f"Error processing image: {e}")
                    # Fall back to text-only
            
            # Text-only generation
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I apologize, but I encountered an error while generating the answer. Please try again."

# Initialize RAG system
rag_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    global rag_system
    try:
        rag_system = RAGSystem()
        logger.info("RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise

@app.post("/api/", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Main API endpoint for RAG queries."""
    try:
        if not rag_system:
            raise HTTPException(status_code=500, detail="RAG system not initialized")
        
        logger.info(f"Processing query: {request.question[:100]}...")
        
        # Search for similar chunks
        similar_chunks = rag_system.search_similar_chunks(request.question, top_k=5)
        
        if not similar_chunks:
            return QueryResponse(
                answer="I couldn't find relevant information in the course materials to answer your question.",
                links=[]
            )
        
        # Generate answer
        answer = rag_system.generate_answer(
            request.question, 
            similar_chunks, 
            request.image
        )
        
        # Extract links from all relevant chunks
        all_links = []
        for chunk in similar_chunks:
            chunk_links = rag_system.extract_links_from_text(chunk['text'])
            all_links.extend(chunk_links)
        
        # Remove duplicates and limit to top 5
        unique_links = []
        seen_urls = set()
        for link in all_links:
            if link['url'] not in seen_urls:
                unique_links.append(LinkResponse(url=link['url'], text=link['text']))
                seen_urls.add(link['url'])
                if len(unique_links) >= 5:
                    break
        
        logger.info(f"Generated answer with {len(unique_links)} links")
        
        return QueryResponse(
            answer=answer,
            links=unique_links
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "embeddings_loaded": rag_system is not None and rag_system.embeddings is not None,
        "num_embeddings": len(rag_system.embeddings) if rag_system and rag_system.embeddings is not None else 0
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Assistant API",
        "endpoints": {
            "POST /api/": "Submit a question with optional image",
            "GET /health": "Health check",
        },
        "example_request": {
            "question": "Should I use gpt-4o-mini which AI proxy supports, or gpt3.5 turbo?",
            "image": "base64_encoded_image_string_optional"
        }
    }

if __name__ == "__main__":
    # For local development
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
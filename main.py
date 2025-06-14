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

# Configure logging for production
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

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
        self._initialized = False
    
    def _lazy_load(self):
        """Lazy load embeddings to reduce cold start time."""
        if self._initialized:
            return
        
        try:
            if not os.path.exists(self.embeddings_file):
                logger.error(f"Embeddings file {self.embeddings_file} not found")
                return
            
            data = np.load(self.embeddings_file, allow_pickle=True)
            self.embeddings = data['embeddings']
            self.metadata = data['metadata']
            self._initialized = True
            logger.info(f"Loaded {len(self.embeddings)} embeddings")
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
    
    def get_query_embedding(self, query: str) -> Optional[np.ndarray]:
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
            return None
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def search_similar_chunks(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find most similar chunks to the query."""
        self._lazy_load()
        
        if not self._initialized or self.embeddings is None:
            return []
        
        query_embedding = self.get_query_embedding(query)
        if query_embedding is None:
            return []
        
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            try:
                similarity = self.cosine_similarity(query_embedding, doc_embedding)
                similarities.append({
                    'index': i,
                    'similarity': similarity,
                    'file': str(self.metadata[i]['file']),
                    'chunk_id': int(self.metadata[i]['chunk_id']),
                    'text': str(self.metadata[i]['text'])[:1000]
                })
            except Exception as e:
                continue
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def extract_links_from_text(self, text: str) -> List[Dict[str, str]]:
        """Extract URLs and their context from text."""
        links = []
        
        # Pattern to match markdown links [text](url)
        markdown_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        matches = re.findall(markdown_pattern, text)
        for text_part, url in matches:
            links.append({"url": url, "text": text_part[:50]})
        
        # Pattern to match plain URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`[\]]+'
        urls = re.findall(url_pattern, text)[:3]
        
        for url in urls:
            url_index = text.find(url)
            start = max(0, url_index - 25)
            end = min(len(text), url_index + len(url) + 25)
            context = text[start:end].strip()
            context = re.sub(r'\s+', ' ', context)
            if len(context) > 50:
                context = context[:47] + "..."
            
            links.append({"url": url, "text": context})
        
        return links[:3]
    
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]], image_data: Optional[str] = None) -> str:
        """Generate answer using Gemini with retrieved context."""
        try:
            # Prepare context
            context = "\n\n".join([
                f"From {chunk['file']}:\n{chunk['text'][:500]}"
                for chunk in context_chunks[:3]
            ])
            
            # Create prompt
            prompt = f"""Answer the student question based on the context.

Context:
{context}

Question: {query}

Answer concisely:"""

            # Handle image if provided
            if image_data:
                try:
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes))
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content([prompt, image])
                    return response.text[:1000]
                except Exception as e:
                    logger.error(f"Error processing image: {e}")
            
            # Text-only generation
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)
            return response.text[:1000]
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I encountered an error while generating the answer. Please try again."

# Initialize RAG system globally
rag_system = RAGSystem()

# Initialize FastAPI app
app = FastAPI(
    title="RAG Assistant", 
    description="RAG-based Q&A system",
    docs_url=None,
    redoc_url=None
)

@app.post("/api/", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Main API endpoint for RAG queries."""
    try:
        # Search for similar chunks
        similar_chunks = rag_system.search_similar_chunks(request.question, top_k=3)
        
        if not similar_chunks:
            return QueryResponse(
                answer="I couldn't find relevant information to answer your question.",
                links=[]
            )
        
        # Generate answer
        answer = rag_system.generate_answer(
            request.question, 
            similar_chunks, 
            request.image
        )
        
        # Extract links from relevant chunks
        all_links = []
        for chunk in similar_chunks[:2]:
            chunk_links = rag_system.extract_links_from_text(chunk['text'])
            all_links.extend(chunk_links)
        
        # Remove duplicates and limit
        unique_links = []
        seen_urls = set()
        for link in all_links:
            if link['url'] not in seen_urls and len(unique_links) < 3:
                unique_links.append(LinkResponse(url=link['url'], text=link['text']))
                seen_urls.add(link['url'])
        
        return QueryResponse(
            answer=answer,
            links=unique_links
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Error processing query")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Assistant API",
        "endpoints": {
            "POST /api/": "Submit a question",
            "GET /health": "Health check"
        }
    }

# This is the key part for Vercel compatibility
from mangum import Mangum
handler = Mangum(app)
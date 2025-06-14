import hashlib
import httpx
import json
import numpy as np
import os
import time
from pathlib import Path
from semantic_text_splitter import MarkdownSplitter
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
def get_chunks(file_path):
    """Extract and chunk markdown content from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Initialize markdown splitter with reasonable chunk size
        splitter = MarkdownSplitter(1000)  # max_characters as first positional argument
        chunks = splitter.chunks(content)
        return chunks
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def get_embedding(text, model="models/text-embedding-004", rate_limit_delay=0.2, max_retries=5):
    """Get embedding for text using Gemini API with exponential backoff retry logic.
    
    Note: Gemini Embedding Experimental has strict limits:
    - Free tier: 5 RPM, 100 RPD 
    - Paid tiers: 10 RPM, 1000 RPD
    Rate limit delay of 12s = 5 requests per minute for free tier
    """
    for attempt in range(max_retries + 1):
        try:
            if attempt == 0:
                time.sleep(rate_limit_delay)  # Rate limiting for embedding API
            else:
                # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                wait_time = 2 ** (attempt - 1)
                print(f"Retry attempt {attempt}, waiting {wait_time}s...")
                time.sleep(wait_time)
            
            result = genai.embed_content(
                model=model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for quota exceeded errors
            if "quota" in error_msg or "limit" in error_msg or "429" in error_msg:
                print(f"Rate/quota limit hit: {e}")
                if "daily" in error_msg or "quota" in error_msg:
                    print("Daily quota likely exceeded. Consider upgrading tier or waiting 24 hours.")
                    return None
                # For RPM limits, wait longer
                wait_time = 60 * (attempt + 1)  # Wait 1, 2, 3... minutes
                print(f"Rate limit hit, waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            
            if attempt == max_retries:
                print(f"Failed after {max_retries} retries. Final error: {e}")
                return None
            else:
                print(f"Attempt {attempt + 1} failed: {e}")
                continue
    
    return None

def load_existing_embeddings(output_file):
    """Load existing embeddings from npz file."""
    if not os.path.exists(output_file):
        return []
    
    try:
        data = np.load(output_file, allow_pickle=True)
        embeddings = data['embeddings']
        metadata = data['metadata']
        
        existing_data = []
        for i, embedding in enumerate(embeddings):
            existing_data.append({
                'file': str(metadata[i]['file']),
                'chunk_id': int(metadata[i]['chunk_id']),
                'text': str(metadata[i]['text']),
                'embedding': embedding
            })
        
        print(f"Loaded {len(existing_data)} existing embeddings from {output_file}")
        return existing_data
    except Exception as e:
        print(f"Error loading existing embeddings: {e}")
        return []

def get_processed_chunks(existing_embeddings):
    """Get set of already processed (file, chunk_id) pairs."""
    processed = set()
    for item in existing_embeddings:
        processed.add((item['file'], item['chunk_id']))
    return processed

def save_embeddings_to_npz(embeddings_data, output_file):
    """Save embeddings and metadata to npz file."""
    embeddings = []
    metadata = []
    
    for item in embeddings_data:
        embeddings.append(item['embedding'])
        metadata.append({
            'file': item['file'],
            'chunk_id': item['chunk_id'],
            'text': item['text']
        })
    
    embeddings_array = np.array(embeddings)
    
    # Save to npz file
    np.savez_compressed(
        output_file,
        embeddings=embeddings_array,
        metadata=metadata
    )
    print(f"Saved {len(embeddings)} embeddings to {output_file}")

if __name__ == "__main__":
    # Configure Gemini API
    # Make sure to set your API key: export GOOGLE_API_KEY="your-api-key"
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    
    output_file = "markdown_embeddings.npz"
    
    # Load existing embeddings
    print("Checking for existing embeddings...")
    existing_embeddings = load_existing_embeddings(output_file)
    processed_chunks = get_processed_chunks(existing_embeddings)
    
    # Find all markdown files
    files = [*Path("markdowns").glob("*.md"), *Path("markdowns").rglob("*.md")]
    
    all_chunks = []
    all_embeddings = existing_embeddings.copy()  # Start with existing embeddings
    total_chunks = 0
    file_chunks = {}
    new_chunks_to_process = []
    
    # First pass: collect all chunks and identify new ones
    print("Collecting chunks from markdown files...")
    for file_path in files:
        chunks = get_chunks(file_path)
        file_chunks[file_path] = chunks
        total_chunks += len(chunks)
        
        # Check which chunks are new
        for chunk_id, chunk in enumerate(chunks):
            chunk_key = (file_path.name, chunk_id)
            if chunk_key not in processed_chunks:
                new_chunks_to_process.append({
                    'file_path': file_path,
                    'chunk_id': chunk_id,
                    'chunk': chunk
                })
    
    print(f"Found {total_chunks} total chunks across {len(files)} files")
    print(f"Already processed: {len(existing_embeddings)} chunks")
    print(f"New chunks to process: {len(new_chunks_to_process)} chunks")
    
    # Debug: Show some examples of processed chunks
    print(f"\nDEBUG: First 10 processed chunk keys:")
    processed_list = list(processed_chunks)[:10]
    for key in processed_list:
        print(f"  {key}")
    
    # Debug: Show some examples of current chunks  
    print(f"\nDEBUG: First 10 current chunk keys:")
    current_keys = []
    count = 0
    for file_path in files:
        chunks = get_chunks(file_path)
        for chunk_id, chunk in enumerate(chunks):
            current_keys.append((file_path.name, chunk_id))
            count += 1
            if count >= 10:
                break
        if count >= 10:
            break
    
    for key in current_keys:
        print(f"  {key}")
        
    # Debug: Check if there's a mismatch
    print(f"\nDEBUG: Example matches:")
    for i, current_key in enumerate(current_keys[:5]):
        is_processed = current_key in processed_chunks
        print(f"  {current_key} -> {'PROCESSED' if is_processed else 'NEW'}")
    
    if len(new_chunks_to_process) == 0:
        print("\nAll chunks already processed!")
        print("This might indicate that the file order or chunk splitting changed.")
        print("If you want to reprocess everything, delete the existing .npz file.")
        exit(0)
    
    # Warning about rate limits
    print("\n⚠️  IMPORTANT: Gemini Embedding API has strict limits:")
    print("   Free tier: 5 requests/minute, 100 requests/day")
    print("   Paid tiers: 10 requests/minute, 1000 requests/day")
    print(f"   With {len(new_chunks_to_process)} remaining chunks, this will take time")
    print("   Consider upgrading your tier at: https://aistudio.google.com/app/apikey\n")
    
    # Process only new chunks with progress bar
    with tqdm(total=len(new_chunks_to_process), desc="Processing new embeddings") as pbar:
        for item in new_chunks_to_process:
            try:
                embedding = get_embedding(item['chunk'])
                if embedding is not None:
                    all_embeddings.append({
                        'file': item['file_path'].name,
                        'chunk_id': item['chunk_id'],
                        'text': item['chunk'],
                        'embedding': embedding
                    })
                    
                    # Save progress every 10 embeddings
                    if len(all_embeddings) % 10 == 0:
                        save_embeddings_to_npz(all_embeddings, output_file)
                
                pbar.set_postfix({
                    "file": item['file_path'].name, 
                    "total": len(all_embeddings),
                    "new": len(all_embeddings) - len(existing_embeddings)
                })
                pbar.update(1)
                
            except Exception as e:
                print(f"Error processing chunk {item['chunk_id']} from {item['file_path']}: {e}")
                pbar.update(1)
                continue
    
    # Final save
    if all_embeddings:
        save_embeddings_to_npz(all_embeddings, output_file)
        print(f"Successfully processed {len(all_embeddings)} total embeddings")
        print(f"Added {len(all_embeddings) - len(existing_embeddings)} new embeddings")
    else:
        print("No embeddings were generated")
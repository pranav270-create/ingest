from src.pipeline.registry import FunctionRegistry
from src.schemas.schemas import Document, Entry, ChunkingMethod
from src.chunking.chunk_utils import document_to_content, chunks_to_entries

@FunctionRegistry.register("chunk", ChunkingMethod.SLIDING_WINDOW.value)
async def sliding_window_chunks(documents: list[Document], **kwargs) -> list[Document]:
    """Chunk documents using sliding window approach."""
    chunk_size = kwargs.get("chunk_size", 1000)
    overlap = kwargs.get("overlap", 200)
    
    chunking_metadata = {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "method": "sliding_window"
    }
    
    new_docs = []
    for doc in documents:
        content = document_to_content(doc)
        chunks = []
        
        # Create overlapping chunks
        text = " ".join(content)
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk:  # Only add non-empty chunks
                chunks.append({"text": chunk, "pages": []})  # Page tracking would need to be implemented
                
        formatted_entries = chunks_to_entries(doc, chunks, ChunkingMethod.SLIDING, chunking_metadata)
        doc.entries = formatted_entries
        new_docs.append(doc)
        
    return new_docs 
import os
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(path="./chroma_db")
        except:
            # Fallback for older ChromaDB versions
            self.client = chromadb.Client()

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection (works with multiple ChromaDB versions)
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "RAG document collection"},
            )
        except AttributeError:
            # Fallback for older ChromaDB versions
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "RAG document collection"},
                )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 250) -> List[str]:
        """
        Simple text chunking by splitting on spaces and grouping into chunks.

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk

        Returns:
            List of text chunks
        """
        self.chunk_splits = RecursiveCharacterTextSplitter(chunk_size= chunk_size, chunk_overlap = 50)
        chunks = self.chunk_splits.split_text(text)
        return chunks

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents
        """
        print(f"Processing {len(documents)} documents...")
        
        all_chunks = []
        all_ids = []
        all_metadatas = []
        
        # Process each document
        for doc_idx, doc in enumerate(documents):
            # Extract content and metadata
            content = doc.get('content', '') if isinstance(doc, dict) else str(doc)
            metadata = doc.get('metadata', {}) if isinstance(doc, dict) else {}
            
            # Chunk the document
            chunks = self.chunk_text(content)
            
            # Create unique IDs and metadata for each chunk
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"doc_{doc_idx}_chunk_{chunk_idx}"
                chunk_metadata = {
                    **metadata,
                    'doc_id': doc_idx,
                    'chunk_id': chunk_idx,
                    'total_chunks': len(chunks)
                }
                
                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                all_metadatas.append(chunk_metadata)
        
        if not all_chunks:
            print("No chunks to add")
            return
        
        # Generate embeddings for all chunks
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)
        
        # Add to ChromaDB collection
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )
        
        print(f"Documents added to vector database ({len(all_chunks)} chunks total)")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        
        # ChromaDB returns results in a specific format, extract them
        return {
            "documents": results.get('documents', [[]])[0],
            "metadatas": results.get('metadatas', [[]])[0],
            "distances": results.get('distances', [[]])[0],
            "ids": results.get('ids', [[]])[0],
        }
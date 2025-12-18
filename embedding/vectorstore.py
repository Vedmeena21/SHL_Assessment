"""
Assessment Vector Store

Manages embeddings and semantic search for SHL assessments using ChromaDB.
"""

import json
import logging
import os
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
import google.generativeai as genai

logger = logging.getLogger(__name__)


class AssessmentVectorStore:
    """
    Vector database for SHL assessments using ChromaDB and Gemini embeddings
    """
    
    def __init__(self, db_path: str = "./chroma_db", gemini_api_key: Optional[str] = None):
        """
        Initialize vector store
        
        Args:
            db_path: Path to ChromaDB persistent storage
            gemini_api_key: Gemini API key for embeddings
        """
        self.db_path = db_path
        
        # Configure Gemini
        api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        genai.configure(api_key=api_key)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="shl_assessments",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initialized vector store at {db_path}")
    
    def prepare_document_text(self, assessment: Dict) -> str:
        """
        Create rich text representation for embedding
        
        Strategy: Combine multiple fields with repetition for important terms
        
        Args:
            assessment: Assessment dictionary
            
        Returns:
            Rich text representation
        """
        # Repeat name for emphasis
        name_repeated = f"{assessment['name']} " * 3
        
        # Build rich document
        doc_parts = [
            f"Assessment Name: {name_repeated}",
            f"Test Type: {assessment.get('test_type', 'Unknown')}",
            f"Categories: {', '.join(assessment.get('categories', []))}",
            f"Description: {assessment.get('description', '')}",
            f"Full Content: {assessment.get('full_content', '')[:1000]}"
        ]
        
        # Add extracted keywords
        keywords = self.extract_keywords(assessment)
        if keywords:
            doc_parts.append(f"Keywords: {', '.join(keywords)}")
        
        return "\n\n".join(doc_parts).strip()
    
    def extract_keywords(self, assessment: Dict) -> List[str]:
        """
        Extract key technical terms and skills from assessment
        
        Args:
            assessment: Assessment dictionary
            
        Returns:
            List of keywords
        """
        import re
        
        text = (
            assessment.get('name', '') + " " +
            assessment.get('description', '') + " " +
            " ".join(assessment.get('categories', []))
        ).lower()
        
        # Common technical skills and roles
        keywords_patterns = [
            r'\b(java|python|javascript|sql|html|css|selenium|react)\b',
            r'\b(programming|coding|development|testing|qa)\b',
            r'\b(sales|marketing|leadership|management|admin)\b',
            r'\b(communication|english|verbal|written)\b',
            r'\b(numerical|reasoning|aptitude|cognitive)\b',
            r'\b(personality|behavioral|opq|motivation)\b'
        ]
        
        keywords = []
        for pattern in keywords_patterns:
            matches = re.findall(pattern, text)
            keywords.extend(matches)
        
        return list(set(keywords))[:10]  # Limit to 10 unique keywords
    
    def index_assessments(self, assessments: List[Dict]) -> int:
        """
        Add all assessments to vector store
        
        Args:
            assessments: List of assessment dictionaries
            
        Returns:
            Number of assessments indexed
        """
        logger.info(f"Indexing {len(assessments)} assessments...")
        
        documents = []
        metadatas = []
        ids = []
        
        for i, assessment in enumerate(assessments):
            # Prepare document text
            doc_text = self.prepare_document_text(assessment)
            documents.append(doc_text)
            
            # Prepare metadata
            metadatas.append({
                "name": assessment['name'],
                "url": assessment['url'],
                "test_type": assessment.get('test_type', 'O'),
                "categories": json.dumps(assessment.get('categories', [])),
                "description": assessment.get('description', '')[:200]
            })
            
            # Generate ID
            ids.append(f"assessment_{i}")
        
        # Add to collection (ChromaDB will handle embedding generation)
        # Note: For Gemini embeddings, we'll need to generate them manually
        # and use add_embeddings() instead
        
        # For now, using ChromaDB's default embeddings
        # TODO: Integrate Gemini embeddings
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Successfully indexed {len(assessments)} assessments")
        return len(assessments)
    
    def search(self, query: str, k: int = 20) -> List[Dict]:
        """
        Semantic search for assessments
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of assessment dictionaries with scores
        """
        # Perform semantic search
        results = self.collection.query(
            query_texts=[query],
            n_results=min(k, self.collection.count())
        )
        
        # Format results
        assessments = []
        
        if results['ids'] and len(results['ids']) > 0:
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i] if 'distances' in results else None
                
                # Convert distance to similarity score (cosine similarity)
                score = 1 - distance if distance is not None else 0.5
                
                assessments.append({
                    'name': metadata['name'],
                    'url': metadata['url'],
                    'test_type': metadata['test_type'],
                    'description': metadata.get('description', ''),
                    'categories': json.loads(metadata.get('categories', '[]')),
                    'score': score
                })
        
        return assessments
    
    def get_count(self) -> int:
        """Get total number of assessments in vector store"""
        return self.collection.count()


def load_assessments_from_file(file_path: str) -> List[Dict]:
    """
    Load assessments from JSON file
    
    Args:
        file_path: Path to assessments JSON file
        
    Returns:
        List of assessment dictionaries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        assessments = json.load(f)
    
    logger.info(f"Loaded {len(assessments)} assessments from {file_path}")
    return assessments


def main():
    """Test vector store functionality"""
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize vector store
    vector_store = AssessmentVectorStore()
    
    # Load assessments
    assessments = load_assessments_from_file('data/assessments.json')
    
    # Index assessments
    count = vector_store.index_assessments(assessments)
    print(f"Indexed {count} assessments")
    
    # Test search
    test_query = "I need Java developers with good communication skills"
    results = vector_store.search(test_query, k=5)
    
    print(f"\nSearch results for: '{test_query}'")
    print("="*60)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['name']}")
        print(f"   Score: {result['score']:.3f}")
        print(f"   Type: {result['test_type']}")
        print()


if __name__ == "__main__":
    main()

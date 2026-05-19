"""
Memory system for TreeQuest Lab.

Stores and retrieves past ideas to promote novelty and avoid duplicates.
Supports simple in-memory storage and vector-based similarity search.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Any
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Single memory entry."""
    
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    access_count: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "created_at": self.created_at,
            "access_count": self.access_count
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        """Create from dictionary."""
        return cls(**data)


class SimpleMemory:
    """
    Simple in-memory storage for ideas.
    
    Uses keyword matching and recency for retrieval.
    Suitable for small-scale experiments without vector databases.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._entries: dict[str, MemoryEntry] = {}
        self._access_order: list[str] = []
        
    def add(
        self,
        content: str,
        metadata: Optional[dict] = None,
        entry_id: Optional[str] = None
    ) -> MemoryEntry:
        """
        Add content to memory.
        
        Args:
            content: Text content (e.g., hypothesis)
            metadata: Additional metadata
            entry_id: Optional custom ID
            
        Returns:
            Created MemoryEntry
        """
        if entry_id is None:
            entry_id = f"mem_{len(self._entries)}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        if entry_id in self._entries:
            logger.warning(f"Entry {entry_id} already exists, updating")
        
        entry = MemoryEntry(
            id=entry_id,
            content=content,
            metadata=metadata or {}
        )
        
        # Enforce max size (remove oldest)
        if len(self._entries) >= self.max_size and entry_id not in self._entries:
            self._remove_oldest()
        
        self._entries[entry_id] = entry
        self._access_order.append(entry_id)
        
        logger.debug(f"Added memory entry {entry_id}")
        return entry
    
    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve entry by ID."""
        entry = self._entries.get(entry_id)
        if entry:
            entry.access_count += 1
        return entry
    
    def search(
        self,
        query: str,
        limit: int = 10,
        keywords: Optional[list[str]] = None
    ) -> list[MemoryEntry]:
        """
        Search memory by keyword matching.
        
        Args:
            query: Search query string
            limit: Maximum results to return
            keywords: Additional keywords to match
            
        Returns:
            List of matching entries (sorted by relevance)
        """
        query_lower = query.lower()
        keyword_set = set(kw.lower() for kw in (keywords or []))
        
        scored_entries = []
        
        for entry in self._entries.values():
            score = 0.0
            content_lower = entry.content.lower()
            
            # Exact match
            if query_lower in content_lower:
                score += 10.0
            
            # Word overlap
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())
            overlap = len(query_words & content_words)
            score += overlap * 2.0
            
            # Keyword match
            if keyword_set:
                keyword_overlap = len(keyword_set & content_words)
                score += keyword_overlap * 3.0
            
            # Recency bonus
            recency_factor = 1.0 / (1 + len(self._entries) - list(self._entries.keys()).index(entry.id))
            score += recency_factor
            
            # Access count penalty (avoid overusing same ideas)
            score -= entry.access_count * 0.5
            
            if score > 0:
                scored_entries.append((score, entry))
        
        # Sort by score descending
        scored_entries.sort(reverse=True, key=lambda x: x[0])
        
        return [entry for _, entry in scored_entries[:limit]]
    
    def find_similar(self, content: str, threshold: float = 0.5) -> list[MemoryEntry]:
        """
        Find similar content (simple text similarity).
        
        Args:
            content: Content to compare against
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar entries
        """
        similar = []
        
        for entry in self._entries.values():
            similarity = self._text_similarity(content, entry.content)
            if similarity >= threshold:
                similar.append((similarity, entry))
        
        similar.sort(reverse=True, key=lambda x: x[0])
        return [entry for _, entry in similar]
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple Jaccard similarity between texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _remove_oldest(self):
        """Remove oldest entry."""
        if self._access_order:
            oldest_id = self._access_order[0]
            del self._entries[oldest_id]
            self._access_order.pop(0)
            logger.debug(f"Removed oldest memory entry {oldest_id}")
    
    def get_all(self) -> list[MemoryEntry]:
        """Get all entries."""
        return list(self._entries.values())
    
    def count(self) -> int:
        """Get number of entries."""
        return len(self._entries)
    
    def clear(self):
        """Clear all entries."""
        self._entries.clear()
        self._access_order.clear()
    
    def save(self, path: str):
        """Save memory to file."""
        data = {
            "max_size": self.max_size,
            "entries": [entry.to_dict() for entry in self._entries.values()]
        }
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Memory saved to {save_path}")
    
    def load(self, path: str):
        """Load memory from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.max_size = data.get("max_size", self.max_size)
        self._entries = {}
        self._access_order = []
        
        for entry_data in data.get("entries", []):
            entry = MemoryEntry.from_dict(entry_data)
            self._entries[entry.id] = entry
            self._access_order.append(entry.id)
        
        logger.info(f"Memory loaded from {path}, {len(self._entries)} entries")


class VectorMemory(SimpleMemory):
    """
    Vector-based memory with embedding similarity search.
    
    Requires sentence-transformers or similar library.
    Falls back to SimpleMemory if embeddings unavailable.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384
    ):
        super().__init__(max_size=max_size)
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self._model = None
        self._try_load_model()
    
    def _try_load_model(self):
        """Try to load embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not available, using keyword-only search")
            self._model = None
    
    def _get_embedding(self, text: str) -> Optional[list[float]]:
        """Get embedding for text."""
        if self._model is None:
            return None
        
        try:
            embedding = self._model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None
    
    def add(
        self,
        content: str,
        metadata: Optional[dict] = None,
        entry_id: Optional[str] = None
    ) -> MemoryEntry:
        """Add content with embedding."""
        entry = super().add(content, metadata, entry_id)
        
        # Compute and store embedding
        embedding = self._get_embedding(content)
        if embedding:
            entry.embedding = embedding
        
        return entry
    
    def search_by_similarity(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.3
    ) -> list[MemoryEntry]:
        """
        Search by embedding similarity.
        
        Args:
            query: Query string
            limit: Maximum results
            threshold: Minimum cosine similarity
            
        Returns:
            List of similar entries
        """
        if self._model is None:
            # Fall back to keyword search
            return self.search(query, limit)
        
        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            return []
        
        import numpy as np
        
        scored_entries = []
        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)
        
        for entry in self._entries.values():
            if entry.embedding is None:
                continue
            
            entry_vec = np.array(entry.embedding)
            entry_norm = np.linalg.norm(entry_vec)
            
            if entry_norm == 0:
                continue
            
            # Cosine similarity
            similarity = np.dot(query_vec, entry_vec) / (query_norm * entry_norm)
            
            if similarity >= threshold:
                scored_entries.append((similarity, entry))
        
        scored_entries.sort(reverse=True, key=lambda x: x[0])
        return [entry for _, entry in scored_entries[:limit]]


class MemoryManager:
    """
    High-level manager for memory operations.
    
    Provides unified interface for storing and retrieving research ideas.
    """
    
    def __init__(
        self,
        use_vector: bool = False,
        max_size: int = 1000,
        storage_path: Optional[str] = None
    ):
        if use_vector:
            self.memory = VectorMemory(max_size=max_size)
        else:
            self.memory = SimpleMemory(max_size=max_size)
        
        self.storage_path = storage_path
        
    def store_idea(
        self,
        hypothesis: str,
        metadata: Optional[dict] = None
    ) -> MemoryEntry:
        """Store a research idea."""
        if metadata is None:
            metadata = {}
        
        metadata["type"] = "hypothesis"
        
        return self.memory.add(
            content=hypothesis,
            metadata=metadata
        )
    
    def find_similar_ideas(
        self,
        hypothesis: str,
        threshold: float = 0.5
    ) -> list[str]:
        """Find similar hypotheses."""
        if isinstance(self.memory, VectorMemory):
            entries = self.memory.search_by_similarity(hypothesis, threshold=threshold)
        else:
            entries = self.memory.find_similar(hypothesis, threshold)
        
        return [entry.content for entry in entries]
    
    def check_novelty(
        self,
        hypothesis: str,
        threshold: float = 0.7
    ) -> tuple[bool, float]:
        """
        Check if hypothesis is novel enough.
        
        Args:
            hypothesis: Hypothesis to check
            threshold: Maximum allowed similarity
            
        Returns:
            Tuple of (is_novel, max_similarity)
        """
        similar = self.find_similar_ideas(hypothesis, threshold=0.0)
        
        if not similar:
            return True, 0.0
        
        # Calculate max similarity
        max_sim = max(
            self.memory._text_similarity(hypothesis, s)
            for s in similar
        )
        
        is_novel = max_sim < threshold
        return is_novel, max_sim
    
    def get_previous_ideas(self, limit: int = 20) -> list[str]:
        """Get list of previous ideas for context."""
        entries = self.memory.get_all()
        # Sort by recency (newest first)
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return [e.content for e in entries[:limit]]
    
    def save(self):
        """Save memory to disk."""
        if self.storage_path:
            self.memory.save(self.storage_path)
    
    def load(self):
        """Load memory from disk."""
        if self.storage_path and Path(self.storage_path).exists():
            self.memory.load(self.storage_path)


if __name__ == "__main__":
    # Test memory system
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing Memory System")
    print("=" * 60)
    
    # Test simple memory
    memory = SimpleMemory(max_size=100)
    
    # Add some entries
    test_ideas = [
        "LoRA rank adaptation improves convergence",
        "Gradient checkpointing trade-offs in small models",
        "Learning rate scheduling strategies",
        "Batch size effects on generalization"
    ]
    
    for idea in test_ideas:
        memory.add(idea, metadata={"source": "test"})
        print(f"✓ Added: {idea[:40]}...")
    
    print(f"\nTotal entries: {memory.count()}")
    
    # Test search
    print("\nSearching for 'LoRA'...")
    results = memory.search("LoRA", limit=5)
    for r in results:
        print(f"  - {r.content} (accessed {r.access_count} times)")
    
    # Test similarity
    print("\nFinding similar to 'learning rate scheduling'...")
    similar = memory.find_similar("learning rate scheduling", threshold=0.3)
    for s in similar:
        sim_score = memory._text_similarity("learning rate scheduling", s.content)
        print(f"  - {s.content} (similarity: {sim_score:.2f})")
    
    # Test novelty check
    manager = MemoryManager()
    manager.memory = memory
    
    new_idea = "Novel optimization technique with AdamW"
    is_novel, max_sim = manager.check_novelty(new_idea, threshold=0.7)
    print(f"\nNovelty check for '{new_idea[:30]}...'")
    print(f"  Is novel: {is_novel}, Max similarity: {max_sim:.2f}")
    
    print("\n✓ Memory tests complete!")

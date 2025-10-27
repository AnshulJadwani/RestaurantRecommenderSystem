import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional
import faiss
import pickle
import os
from tqdm import tqdm

class RestaurantEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the RestaurantEmbedder with a specific SBERT model."""
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.index = None
        self.restaurant_ids = None

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        embeddings = []
        
        # Use tqdm for progress bar
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch)
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build a FAISS index from embeddings."""
        dimension = embeddings.shape[1]
        
        # Initialize FAISS index
        index = faiss.IndexFlatL2(dimension)
        
        # Add vectors to the index
        embeddings = embeddings.astype('float32')
        index.add(embeddings)
        
        return index

    def save_embeddings(self, save_dir: str, embeddings: np.ndarray, 
                       restaurant_ids: List[int]) -> None:
        """Save embeddings and restaurant IDs to disk."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save embeddings
        np.save(os.path.join(save_dir, 'embeddings.npy'), embeddings)
        
        # Save restaurant IDs
        with open(os.path.join(save_dir, 'restaurant_ids.pkl'), 'wb') as f:
            pickle.dump(restaurant_ids, f)
        
        # Save FAISS index
        index = self.build_faiss_index(embeddings)
        faiss.write_index(index, os.path.join(save_dir, 'restaurants.index'))

    def load_embeddings(self, save_dir: str) -> Tuple[Optional[np.ndarray], 
                                                     Optional[List[int]], 
                                                     Optional[faiss.Index]]:
        """Load embeddings, restaurant IDs, and FAISS index from disk."""
        try:
            # Load embeddings
            embeddings = np.load(os.path.join(save_dir, 'embeddings.npy'))
            
            # Load restaurant IDs
            with open(os.path.join(save_dir, 'restaurant_ids.pkl'), 'rb') as f:
                restaurant_ids = pickle.load(f)
            
            # Load FAISS index
            index = faiss.read_index(os.path.join(save_dir, 'restaurants.index'))
            
            return embeddings, restaurant_ids, index
        except Exception as e:
            print(f"Error loading embeddings: {str(e)}")
            return None, None, None

if __name__ == "__main__":
    # Test the embedder
    embedder = RestaurantEmbedder()
    test_texts = [
        "Italian restaurant with pizza and pasta",
        "Japanese sushi restaurant",
        "Mexican taco place"
    ]
    embeddings = embedder.generate_embeddings(test_texts)
    print("Embeddings shape:", embeddings.shape)
    
    # Test saving and loading
    embedder.save_embeddings('models', embeddings, [1, 2, 3])
    loaded_emb, loaded_ids, loaded_index = embedder.load_embeddings('models')
    print("\nLoaded embeddings shape:", loaded_emb.shape if loaded_emb is not None else None)
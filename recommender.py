import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from embed import RestaurantEmbedder
from preprocess import TextPreprocessor
from summarizer import RestaurantSummarizer
from nlp_simple_aspects import SimpleAspectAnalyzer

class RestaurantRecommender:
    def __init__(self, data: pd.DataFrame, embedder: RestaurantEmbedder, 
                 preprocessor: TextPreprocessor):
        """Initialize the recommender system."""
        self.data = data
        self.embedder = embedder
        self.preprocessor = preprocessor
        self.summarizer = RestaurantSummarizer()
        # lightweight aspect analyzer (VADER-based)
        try:
            self.simple_aspect_analyzer = SimpleAspectAnalyzer()
        except Exception:
            self.simple_aspect_analyzer = None
        self.embeddings = None
        self.index = None
        self.restaurant_ids = None

    def load_or_create_embeddings(self, save_dir: str = 'models', force_create: bool = False) -> None:
        """Load existing embeddings or create new ones."""
        if not force_create:
            # Try to load existing embeddings
            embeddings, restaurant_ids, index = self.embedder.load_embeddings(save_dir)
            if all(x is not None for x in [embeddings, restaurant_ids, index]):
                self.embeddings = embeddings
                self.restaurant_ids = restaurant_ids
                self.index = index
                return

        # Create new embeddings if loading failed or force_create is True
        print("Generating new embeddings...")
        
        # Combine text features for embedding
        combined_texts = []
        restaurant_ids = []
        
        for idx, row in self.data.iterrows():
            combined_text = self.preprocessor.combine_features(
                name=row.get('name', ''),
                cuisine=row.get('cuisine', ''),
                description=row.get('description', ''),
                reviews=row.get('reviews', '')
            )
            combined_texts.append(combined_text)
            restaurant_ids.append(idx)

        # Generate embeddings
        embeddings = self.embedder.generate_embeddings(combined_texts)
        
        # Build FAISS index
        index = self.embedder.build_faiss_index(embeddings)
        
        # Save everything
        self.embedder.save_embeddings(save_dir, embeddings, restaurant_ids)
        
        # Update instance variables
        self.embeddings = embeddings
        self.restaurant_ids = restaurant_ids
        self.index = index

    def recommend_restaurants(self, city: str, cuisine: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Recommend restaurants based on city and cuisine."""
        if self.embeddings is None or self.index is None:
            raise ValueError("Embeddings not initialized. Call load_or_create_embeddings first.")

        # Filter by city and cuisine (case-insensitive)
        city_mask = self.data['city'].str.lower() == city.lower()
        cuisine_mask = self.data['cuisine'].str.lower() == cuisine.lower()
        filtered_restaurants = self.data[city_mask & cuisine_mask]
        
        # If no exact cuisine match, try finding similar cuisines in the specified city
        if len(filtered_restaurants) < top_k:
            # Get all restaurants in the city
            city_restaurants = self.data[city_mask]
            
            if len(city_restaurants) == 0:
                return []

            # Create query embedding for the cuisine
            query_text = f"Restaurant serving {cuisine} cuisine with {cuisine} dishes and {cuisine} flavors"
            query_embedding = self.embedder.model.encode([query_text])[0]
            
            # Convert to correct format for FAISS
            query_embedding = np.array([query_embedding]).astype('float32')
            
            # Get city restaurant indices
            city_indices = city_restaurants.index.tolist()
            
            # If using FAISS
            if isinstance(self.index, faiss.Index):
                # Search in FAISS index for more restaurants than needed to ensure diversity
                D, I = self.index.search(query_embedding, len(city_restaurants))
                
                # Filter results to only include restaurants in the specified city
                filtered_results = [
                    i for i in I[0] 
                    if self.restaurant_ids[i] in city_indices
                ]
                
                # Combine exact matches with similar restaurants
                exact_match_indices = filtered_restaurants.index.tolist()
                # Further filter similar results to ensure they match the requested cuisine
                similar_indices = []
                for i in filtered_results:
                    rid = self.restaurant_ids[i]
                    # Skip if it's already an exact match
                    if rid in exact_match_indices:
                        continue
                    # Ensure cuisine matches (case-insensitive)
                    try:
                        r_cuisine = str(self.data.loc[rid, 'cuisine'])
                    except Exception:
                        r_cuisine = ''
                    if r_cuisine.lower() == cuisine.lower():
                        similar_indices.append(rid)
                
                # Combine exact matches with similar restaurants
                recommended_indices = exact_match_indices + similar_indices[:top_k - len(exact_match_indices)]
            else:
                # Fallback to cosine similarity
                city_embeddings = self.embeddings[city_indices]
                similarities = cosine_similarity(query_embedding, city_embeddings)[0]
                
                # Get indices of most similar restaurants and filter by cuisine
                similar_indices = []
                for i in similarities.argsort()[::-1]:
                    rid = city_indices[i]
                    if rid in filtered_restaurants.index:
                        continue
                    try:
                        r_cuisine = str(self.data.loc[rid, 'cuisine'])
                    except Exception:
                        r_cuisine = ''
                    if r_cuisine.lower() == cuisine.lower():
                        similar_indices.append(rid)
                
                # Combine exact matches with similar restaurants
                recommended_indices = filtered_restaurants.index.tolist() + similar_indices[:top_k - len(filtered_restaurants)]
        else:
            # If we have enough exact matches, use those
            recommended_indices = filtered_restaurants.index.tolist()[:top_k]

        # Sort the recommended indices so the highest-rated (and most voted) restaurants come first
        def _score(idx_val: int):
            try:
                rating_val = float(self.data.at[idx_val, 'rating']) if 'rating' in self.data.columns else 0.0
            except Exception:
                rating_val = 0.0
            try:
                votes_val = int(self.data.at[idx_val, 'votes']) if 'votes' in self.data.columns else 0
            except Exception:
                votes_val = 0
            return (rating_val, votes_val)

        # Remove duplicates while preserving order, then sort by score (rating, votes)
        unique_indices = []
        seen = set()
        for i in recommended_indices:
            if i not in seen:
                seen.add(i)
                unique_indices.append(i)

        sorted_indices = sorted(unique_indices, key=lambda i: _score(i), reverse=True)

        # Trim to requested top_k
        recommended_indices = sorted_indices[:top_k]

        # Prepare recommendations
        recommendations = []
        for idx in recommended_indices:
            restaurant = self.data.loc[idx]
            # Create restaurant dict with all available fields
            restaurant_dict = {
                'id': idx,
                'name': restaurant.get('name', ''),
                'cuisine': restaurant.get('cuisine', ''),
                'city': restaurant.get('city', ''),
                'address': restaurant.get('address', ''),
                'locality': restaurant.get('locality', ''),
                'rating': restaurant.get('rating', 0.0),
                'description': restaurant.get('description', ''),
                'price_range': restaurant.get('price_range', 3),
                'avg_cost': restaurant.get('average_cost_for_two', 0),
                'currency': restaurant.get('currency', '₹'),
                'votes': restaurant.get('votes', 0),
                'has_table_booking': restaurant.get('has_table_booking', 'No'),
                'has_online_delivery': restaurant.get('has_online_delivery', 'No')
            }
            # Generate summary using the summarizer
            restaurant_dict['summary'] = self.summarizer.generate_summary(restaurant_dict)
            # Add lightweight pros/cons from reviews/description if available
            review_text = restaurant.get('reviews', '') or restaurant.get('description', '')
            if self.simple_aspect_analyzer and review_text:
                try:
                    ac = self.simple_aspect_analyzer.extract_pros_cons(str(review_text), top_n=5)
                    restaurant_dict['pros'] = ac.get('pros', [])
                    restaurant_dict['cons'] = ac.get('cons', [])
                    restaurant_dict['aspect_sentiments'] = ac.get('aspect_sentiments', {})
                except Exception:
                    restaurant_dict['pros'] = []
                    restaurant_dict['cons'] = []
                    restaurant_dict['aspect_sentiments'] = {}
            else:
                restaurant_dict['pros'] = []
                restaurant_dict['cons'] = []
                restaurant_dict['aspect_sentiments'] = {}

            recommendations.append(restaurant_dict)

        return recommendations

if __name__ == "__main__":
    # Test the recommender
    from data_loader import DataLoader
    
    # Initialize components
    loader = DataLoader("data/Dataset.csv")
    data = loader.load_data()
    embedder = RestaurantEmbedder()
    preprocessor = TextPreprocessor()
    
    # Create recommender
    recommender = RestaurantRecommender(data, embedder, preprocessor)
    recommender.load_or_create_embeddings()
    
    # Test recommendations
    recommendations = recommender.recommend_restaurants("New York", "Italian")
    print("\nRecommendations:", recommendations)
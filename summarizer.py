import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import Dict, Any, Optional

class RestaurantSummarizer:
    def __init__(self):
        """Initialize the summarizer."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))

    def _get_price_description(self, price_range: int, avg_cost: float, currency: str) -> str:
        """Convert price range and cost to descriptive text."""
        price_map = {
            1: "budget-friendly",
            2: "casual dining",
            3: "upscale",
            4: "fine dining",
            5: "luxury"
        }
        price_desc = price_map.get(price_range, "moderately priced")
        
        def _currency_symbol(curr: str) -> str:
            """Map common currency names to symbols or clean up dataset values."""
            if not curr:
                return ""
            c = curr.lower()
            if "rupee" in c or "rs." in c or "rs" in c:
                return "₹"
            if "pula" in c:
                return "P"
            if "dollar" in c or "usd" in c:
                return "$"
            if "euro" in c:
                return "€"
            if "pound" in c or "gbp" in c:
                return "£"
            # If the dataset already provides a short symbol in parentheses like "Botswana Pula(P)",
            # try to extract the parenthetical content.
            import re
            m = re.search(r"\(([^)]+)\)", curr)
            if m:
                return m.group(1)
            # Fallback to the raw currency string (cleaned)
            return curr

        symbol = _currency_symbol(str(currency))
        if avg_cost > 0:
            # Format cost with no decimals and thousand separators
            try:
                cost_str = f"{int(avg_cost):,}"
            except Exception:
                cost_str = str(avg_cost)
            if symbol:
                return f"{price_desc} (average {symbol} {cost_str} for two)"
            return f"{price_desc} (average {cost_str} for two)"
        return price_desc

    def _get_rating_description(self, rating: float) -> str:
        """Convert rating to descriptive text."""
        if rating >= 4.5:
            return "highly-rated"
        elif rating >= 4.0:
            return "well-rated"
        elif rating >= 3.5:
            return "decently rated"
        else:
            return "rated"

    def _format_cuisine(self, cuisine: str) -> str:
        """Format cuisine string for natural language."""
        if ',' in cuisine:
            cuisines = [c.strip() for c in cuisine.split(',')]
            if len(cuisines) == 2:
                return f"{cuisines[0]} and {cuisines[1]}"
            elif len(cuisines) > 2:
                return f"{cuisines[0]}, {cuisines[1]} and {cuisines[2]}"
        return cuisine

    def generate_summary(self, restaurant_data: dict) -> str:
        """Generate a natural summary of the restaurant."""
        name = restaurant_data.get('name', '')
        cuisine = restaurant_data.get('cuisine', 'Various')
        price_range = restaurant_data.get('price_range', 2)
        avg_cost = restaurant_data.get('avg_cost', 0)
        rating = restaurant_data.get('rating', 0)
        locality = restaurant_data.get('locality', '')
        address = restaurant_data.get('address', '')
        currency = restaurant_data.get('currency', '₹')
        
        # Generate price description
        price_desc = self._get_price_description(price_range, avg_cost, currency)
        rating_text = f"{rating}/5" if rating > 0 else "Not yet rated"
        
        # Create location description
        location_desc = ""
        if locality and address:
            location_desc = f", situated in {locality}"
        
        # Build the summary
        summary = f"{name} is a {price_desc} {cuisine} restaurant{location_desc}. "
        summary += f"It has a rating of {rating_text}."
        
        # Add address if available and different from locality
        if address and not address.startswith(locality):
            summary += f" You can find it at {address}."
        
        return summary

if __name__ == "__main__":
    # Test the summarizer
    test_restaurant = {
        "name": "Izakaya Kikufuji",
        "cuisine": "Japanese",
        "city": "Makati City",
        "rating": 4.5,
        "price_range": 3,
        "average_cost_for_two": 1200,
        "votes": 591,
        "has_table_booking": "Yes",
        "has_online_delivery": "No",
        "description": "Authentic Japanese cuisine in a traditional setting."
    }
    
    summarizer = RestaurantSummarizer()
    print(summarizer.generate_summary(test_restaurant))
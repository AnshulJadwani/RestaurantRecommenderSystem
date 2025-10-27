import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List, Optional

class TextPreprocessor:
    def __init__(self):
        """Initialize the TextPreprocessor."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        """Clean and normalize text data."""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text."""
        if not text:
            return ""

        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)

    def prepare_text_for_embedding(self, texts: List[str]) -> List[str]:
        """Prepare a list of texts for embedding."""
        processed_texts = []
        for text in texts:
            # Clean the text
            cleaned_text = self.clean_text(text)
            # Remove stopwords
            processed_text = self.remove_stopwords(cleaned_text)
            processed_texts.append(processed_text)
        return processed_texts

    def combine_features(self, name: str, cuisine: str, description: str, 
                        reviews: Optional[str] = None) -> str:
        """Combine different text features into a single string."""
        features = [name, cuisine, description]
        if reviews:
            features.append(reviews)
        
        # Clean and combine all features
        cleaned_features = [self.clean_text(str(feature)) for feature in features if feature]
        return " ".join(cleaned_features)

if __name__ == "__main__":
    # Test the TextPreprocessor
    preprocessor = TextPreprocessor()
    test_text = "This is a Test Restaurant! With great Italian food & amazing service. (2023)"
    print("Original text:", test_text)
    print("Cleaned text:", preprocessor.clean_text(test_text))
    print("Without stopwords:", preprocessor.remove_stopwords(preprocessor.clean_text(test_text)))
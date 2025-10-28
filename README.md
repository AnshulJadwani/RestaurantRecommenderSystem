# Restaurant Recommender System

A sophisticated restaurant recommendation system that uses NLP embeddings (Sentence-BERT) to suggest restaurants based on city and cuisine preferences. The system processes text data from restaurant names, cuisines, descriptions, and reviews to create meaningful recommendations.

## Features

- NLP-powered recommendations using Sentence-BERT embeddings
- Fast similarity search using FAISS (with sklearn cosine similarity fallback)
- City and cuisine-based filtering
- Interactive Streamlit UI
- Persistent storage of embeddings for quick startup
- Robust text preprocessing and cleaning
- Graceful handling of missing or inconsistent data
 - Aspect-based pros/cons extraction from reviews and descriptions (KeyBERT + sentiment)
 - Improved natural-language summaries with formatted currency symbols

## Project Structure

```
RestaurantRecommenderSystem/
├── data/
│   └── Dataset.csv
├── models/
│   ├── embeddings.npy
│   ├── restaurant_ids.pkl
│   └── restaurants.index
├── app.py
├── data_loader.py
├── preprocess.py
├── embed.py
├── recommender.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/RestaurantRecommenderSystem.git
cd RestaurantRecommenderSystem
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

If you plan to use the aspect-extraction and sentiment features (recommended for richer NLP demos), also install the heavier NLP packages:

```bash
pip install transformers torch keybert
```

4. Place your dataset:
Place your `Dataset.csv` file in the `data/` directory.

## Dataset Format

The system expects a CSV file with the following columns:
- name: Restaurant name
- city: Location of the restaurant
- cuisine: Type of cuisine
- rating: Numerical rating
- description: Restaurant description
- reviews: Customer reviews (optional)

Notes about fields present in the included dataset (Zomato-style CSV):
- `Restaurant Name` -> normalized to `name`
- `Cuisines` -> normalized to `cuisine`
- `Average Cost for two` -> normalized to `average_cost_for_two` and used for price text
- `Currency` -> dataset string such as "Indian Rupees(Rs.)"; the app maps these to symbols (₹, $, €, etc.)

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the sidebar to:
   - Select a city
   - Choose a cuisine type
   - Click "Find Restaurants" to get recommendations

## Components

### data_loader.py
- Handles data loading and initial cleaning
- Manages missing values and data consistency

### preprocess.py
- Text cleaning and normalization
- Stopword removal
- Feature combination for embedding

### embed.py
- Generates embeddings using Sentence-BERT
- Manages FAISS index creation and persistence
- Handles embedding storage and retrieval

### recommender.py
- Implements the core recommendation logic
- Manages city and cuisine filtering
- Provides ranked recommendations
 - Integrates the summarizer and aspect sentiment analyzer to attach natural summaries and pros/cons to results

### app.py
- Streamlit-based user interface
- Interactive filters and results display
- Cached loading for better performance
 - Displays generated summaries and aspect-based pros/cons under each restaurant card

## Performance Considerations

- Embeddings are computed once and saved to disk
- FAISS index enables fast similarity search
- Fallback to sklearn's cosine similarity if FAISS fails
- Caching of heavy computations in the Streamlit app
 - Aspect extraction and sentiment use transformer models; enable them for demos but expect higher memory/latency. Consider running with smaller models or enabling them only when needed.

## Error Handling

The system includes robust error handling for:
- Missing or corrupted data files
- Invalid input combinations
- Missing values in the dataset
- Failed embedding generation
- Search/recommendation failures

Additional notes:
- If transformers/KeyBERT are not installed, aspect extraction will fall back gracefully and the app will still return recommendations and rule-based summaries.
- The summarizer maps common currency strings to symbols so summaries show properly formatted amounts (e.g., ₹ 250).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Sentence-BERT for text embeddings
- FAISS for efficient similarity search
- Streamlit for the user interface
- NLTK for text preprocessing
 - KeyBERT and Hugging Face Transformers for aspect extraction and sentiment
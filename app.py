import streamlit as st
import pandas as pd
from data_loader import DataLoader
from preprocess import TextPreprocessor
from embed import RestaurantEmbedder
from recommender import RestaurantRecommender

# Set page config
st.set_page_config(
    page_title="Restaurant Recommender System",
    page_icon="🍽️",
    layout="wide"
)

@st.cache_resource
def load_recommender():
    """Load and initialize the recommender system."""
    # Initialize components
    loader = DataLoader("data/Dataset.csv")
    data = loader.load_data()
    embedder = RestaurantEmbedder()
    preprocessor = TextPreprocessor()
    
    # Create and initialize recommender
    recommender = RestaurantRecommender(data, embedder, preprocessor)
    recommender.load_or_create_embeddings()
    
    return recommender, data

def main():
    # Add title and description
    st.title("🍽️ Restaurant Recommender System")
    st.write("Get personalized restaurant recommendations based on city and cuisine!")
    
    try:
        # Load recommender system
        with st.spinner("Loading recommender system..."):
            recommender, data = load_recommender()

        # Center the input form
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.write("### Tell us what you're looking for! 🌟")
            
            # Get unique cities and cuisines for suggestions
            cities = sorted(data['city'].unique().tolist())
            cuisines = sorted(data['cuisine'].unique().tolist())
            
            # Show available cities and cuisines as helper text
            with st.expander("See available cities"):
                st.write(", ".join(cities))
            with st.expander("See available cuisines"):
                st.write(", ".join(cuisines))
            
            # Text input fields
            selected_city = st.text_input("Enter your city name:", 
                                        placeholder="e.g., New York, London, Tokyo")
            selected_cuisine = st.text_input("What cuisine would you like to try?",
                                          placeholder="e.g., Italian, Japanese, Indian")
            
            # Add search button
            if st.button("🔍 Find My Perfect Restaurant!", use_container_width=True):
                # Validate inputs
                if not selected_city or not selected_cuisine:
                    st.warning("Please enter both city and cuisine! 🙏")
                    return
                
                # Convert inputs to title case for matching
                selected_city = selected_city.strip().title()
                selected_cuisine = selected_cuisine.strip().title()
                
                # Validate city
                if selected_city not in cities:
                    st.error(f"Sorry, we don't have restaurants in {selected_city} yet! 😢")
                    st.write("Available cities:", ", ".join(cities))
                    return
                
                # Validate cuisine
                if selected_cuisine not in cuisines:
                    st.error(f"Sorry, we don't have {selected_cuisine} restaurants in our database! 🍽️")
                    st.write("Available cuisines:", ", ".join(cuisines))
                    return
            # Get recommendations
            with st.spinner("Finding the best restaurants for you..."):
                recommendations = recommender.recommend_restaurants(
                    city=selected_city,
                    cuisine=selected_cuisine,
                    top_k=5
                )

            if recommendations:
                # Display recommendations in cards
                st.balloons()  # Add a fun animation
                st.success(f"🎉 Found these amazing {selected_cuisine} restaurants in {selected_city} for you!")
                
                for i, restaurant in enumerate(recommendations, 1):
                    with st.container():
                        st.write(f"### {i}. {restaurant['name']} {'⭐' * int(round(restaurant['rating']))}")
                        
                        # Create three columns for restaurant details
                        info_col1, info_col2 = st.columns([1, 2])
                        
                        with info_col1:
                            st.write(f"**Location:** {restaurant['city']}")
                            st.write(f"**Cuisine:** {restaurant['cuisine']}")
                            st.write(f"**Rating:** {restaurant['rating']:.1f}/5.0")
                        
                        with info_col2:
                            if restaurant['description']:
                                st.write("**About this place:**")
                                st.info(restaurant['description'])
                        
                        # Add a "Make Reservation" button (placeholder functionality)
                        if st.button(f"🪑 Make Reservation at {restaurant['name']}", key=f"reserve_{i}"):
                            st.write("🎉 This is a demo! In a real app, this would connect to a reservation system.")
                        
                        st.divider()
            else:
                st.warning(f"😢 We couldn't find any {selected_cuisine} restaurants in {selected_city}.")
                st.write("Try another cuisine or city from the available options shown above.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please make sure the dataset is available in the data folder.")

if __name__ == "__main__":
    main()
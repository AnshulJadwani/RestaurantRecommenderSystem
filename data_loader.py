import pandas as pd
import numpy as np
from typing import Dict, Any

class DataLoader:
    def __init__(self, file_path: str):
        """Initialize DataLoader with the path to the dataset."""
        self.file_path = file_path
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """Load the dataset and perform initial cleaning."""
        try:
            self.data = pd.read_csv(self.file_path)
            return self.clean_data()
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return pd.DataFrame()

    def clean_data(self) -> pd.DataFrame:
        """Clean and preprocess the dataset."""
        if self.data is None:
            return pd.DataFrame()

        # Make a copy to avoid modifying the original data
        df = self.data.copy()

        # Convert column names to lowercase and replace spaces with underscores
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]

        # Normalize common name column variants to 'name'
        if 'restaurant_name' in df.columns:
            df = df.rename(columns={'restaurant_name': 'name'})
        if 'restaurant' in df.columns and 'name' not in df.columns:
            df = df.rename(columns={'restaurant': 'name'})

        # Rename 'cuisines' to 'cuisine' for consistency
        if 'cuisines' in df.columns:
            df = df.rename(columns={'cuisines': 'cuisine'})

        # Handle missing values
        text_columns = ['name', 'cuisine', 'description', 'reviews']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('')

        # Convert city names to title case for consistency
        if 'city' in df.columns:
            df['city'] = df['city'].str.title()

        # Clean up cuisine data
        if 'cuisine' in df.columns:
            # Split multiple cuisines and take the first one for simplicity
            df['cuisine'] = df['cuisine'].apply(lambda x: str(x).split(',')[0].strip())
            df['cuisine'] = df['cuisine'].replace('', 'International')
            df['cuisine'] = df['cuisine'].fillna('International')

        # Handle missing numerical values
        if 'aggregate_rating' in df.columns:
            df = df.rename(columns={'aggregate_rating': 'rating'})
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            df['rating'] = df['rating'].fillna(df['rating'].mean())

        return df

    def get_unique_cities(self) -> list:
        """Return list of unique cities in the dataset."""
        if self.data is not None and 'city' in self.data.columns:
            return sorted(self.data['city'].unique().tolist())
        return []

    def get_unique_cuisines(self) -> list:
        """Return list of unique cuisines in the dataset."""
        if self.data is not None and 'cuisine' in self.data.columns:
            return sorted(self.data['cuisine'].unique().tolist())
        return []

if __name__ == "__main__":
    # Test the DataLoader
    loader = DataLoader("data/Dataset.csv")
    df = loader.load_data()
    print("Dataset shape:", df.shape)
    print("\nUnique cities:", loader.get_unique_cities())
    print("\nUnique cuisines:", loader.get_unique_cuisines())
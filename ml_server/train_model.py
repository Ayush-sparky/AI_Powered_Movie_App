import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import requests
import zipfile
import os
import json
import warnings
import gc
import psutil
warnings.filterwarnings('ignore')

class MovieRecommendationModel:
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.user_movie_matrix = None
        self.frequent_itemsets = None
        self.association_rules_df = None
        self.movie_recommendations = {}
        
    def check_memory(self):
        """Monitor memory usage"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        print(f"Memory usage: {memory_mb:.2f} MB")
        return memory_mb
        
    def download_dataset(self):
        """Download and extract MovieLens 100K dataset"""
        print("Downloading MovieLens 100K dataset...")
        
        # Create data directory
        os.makedirs('data', exist_ok=True)
        
        # Download the dataset
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        
        try:
            response = requests.get(url)
            with open('data/ml-100k.zip', 'wb') as f:
                f.write(response.content)
            
            # Extract the zip file
            with zipfile.ZipFile('data/ml-100k.zip', 'r') as zip_ref:
                zip_ref.extractall('data/')
            
            print("Dataset downloaded and extracted successfully!")
            return True
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please manually download from: https://files.grouplens.org/datasets/movielens/ml-100k.zip")
            return False
    
    def load_data_efficiently(self, sample_fraction=0.8):
        """Load the MovieLens data with memory optimization"""
        print("Loading dataset efficiently...")
        
        # Load ratings data with dtype optimization
        ratings_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
        self.ratings_df = pd.read_csv('data/ml-100k/u.data', 
                                     sep='\t', 
                                     names=ratings_columns,
                                     dtype={'user_id': 'int16', 'movie_id': 'int16', 
                                           'rating': 'int8', 'timestamp': 'int32'})
        
        # Sample the data to reduce memory usage
        if sample_fraction < 1.0:
            self.ratings_df = self.ratings_df.sample(frac=sample_fraction, random_state=42)
            print(f"Sampled {sample_fraction*100}% of ratings data")
        
        # Load only essential movie data
        movies_columns = ['movie_id', 'title', 'release_date', 'video_release_date',
                         'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
                         'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                         'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                         'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        
        self.movies_df = pd.read_csv('data/ml-100k/u.item', 
                                    sep='|', 
                                    names=movies_columns, 
                                    encoding='latin-1',
                                    usecols=['movie_id', 'title'])  # Only load essential columns
        
        # Optimize data types
        self.movies_df['movie_id'] = self.movies_df['movie_id'].astype('int16')
        
        print(f"Loaded {len(self.ratings_df)} ratings for {len(self.movies_df)} movies")
        print(f"Number of users: {self.ratings_df['user_id'].nunique()}")
        self.check_memory()
        
    def prepare_transaction_data_optimized(self, min_rating=3.5, min_movie_frequency=20, max_movies_per_user=50):
        """
        Convert ratings data to transaction format with balanced filtering
        """
        print(f"Preparing transaction data (ratings >= {min_rating})...")
        
        # Filter for good ratings (not just excellent ones)
        high_ratings = self.ratings_df[self.ratings_df['rating'] >= min_rating].copy()
        print(f"Good ratings count: {len(high_ratings)}")
        
        # Less aggressive movie filtering
        movie_counts = high_ratings['movie_id'].value_counts()
        popular_movies = movie_counts[movie_counts >= min_movie_frequency].index
        
        print(f"Filtering movies: keeping {len(popular_movies)} out of {movie_counts.nunique()} movies")
        print(f"Movies must appear in at least {min_movie_frequency} user transactions")
        
        # Keep only popular movies
        filtered_ratings = high_ratings[high_ratings['movie_id'].isin(popular_movies)].copy()
        
        # Filter users reasonably
        user_movie_counts = filtered_ratings['user_id'].value_counts()
        # Keep users with at least 3 movies and at most max_movies_per_user
        reasonable_users = user_movie_counts[
            (user_movie_counts >= 3) & (user_movie_counts <= max_movies_per_user)
        ].index
        filtered_ratings = filtered_ratings[filtered_ratings['user_id'].isin(reasonable_users)].copy()
        
        print(f"Filtered to {len(reasonable_users)} users with 3-{max_movies_per_user} movies each")
        
        # Clean up memory
        del high_ratings, movie_counts, user_movie_counts
        gc.collect()
        
        # Group by user to create transactions (baskets)
        user_movies = filtered_ratings.groupby('user_id')['movie_id'].apply(list).reset_index()
        user_movies.columns = ['user_id', 'movies']
        
        # Convert movie IDs to movie titles for better readability
        movie_id_to_title = dict(zip(self.movies_df['movie_id'], self.movies_df['title']))
        
        transactions = []
        for _, row in user_movies.iterrows():
            movie_titles = [movie_id_to_title.get(movie_id, f"Movie_{movie_id}") 
                           for movie_id in row['movies']]
            # Include transactions with at least 2 movies
            if len(movie_titles) >= 2:
                transactions.append(movie_titles)
        
        # Clean up memory
        del user_movies, filtered_ratings, movie_id_to_title
        gc.collect()
        
        print(f"Created {len(transactions)} transactions from user watch history")
        print(f"Average movies per transaction: {np.mean([len(t) for t in transactions]):.2f}")
        print(f"Unique movies in transactions: {len(set([movie for transaction in transactions for movie in transaction]))}")
        
        self.check_memory()
        return transactions
    
    def apply_fpgrowth_algorithm(self, transactions, min_support=0.05):
        """Apply FP-Growth algorithm with adaptive parameters"""
        print("Applying FP-Growth algorithm (memory efficient)...")
        
        # Convert transactions to binary matrix format
        print("Converting transactions to binary matrix...")
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        
        print(f"Transaction matrix shape: {df.shape}")
        print(f"Matrix memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        print(f"Using minimum support: {min_support}")
        
        self.check_memory()
        
        # Use FP-Growth with adaptive support
        frequent_itemsets = None
        current_support = min_support
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                print(f"Attempt {attempt + 1}: trying min_support = {current_support:.3f}")
                frequent_itemsets = fpgrowth(df, min_support=current_support, use_colnames=True)
                
                if len(frequent_itemsets) >= 20:  # We want at least 20 frequent itemsets
                    print(f"Success! Found {len(frequent_itemsets)} frequent itemsets")
                    break
                else:
                    print(f"Only found {len(frequent_itemsets)} itemsets, trying lower support...")
                    current_support = current_support * 0.7  # Reduce support
                    
            except Exception as e:
                print(f"FP-Growth failed with support {current_support:.3f}: {e}")
                current_support = current_support * 0.7
        
        # Clean up the transaction matrix
        del df, te_ary, te
        gc.collect()
        
        if frequent_itemsets is None or len(frequent_itemsets) == 0:
            print("No frequent itemsets found even with low support.")
            return None, None
        
        print(f"Final: Found {len(frequent_itemsets)} frequent itemsets with support >= {current_support:.3f}")
        
        # Generate association rules with multiple attempts
        rules = None
        confidence_thresholds = [0.3, 0.2, 0.15, 0.1, 0.05]
        
        for threshold in confidence_thresholds:
            try:
                print(f"Trying to generate rules with confidence >= {threshold}")
                rules = association_rules(frequent_itemsets, 
                                        metric="confidence", 
                                        min_threshold=threshold)
                
                if len(rules) >= 10:  # We want at least 10 rules
                    print(f"Success! Generated {len(rules)} association rules")
                    break
                else:
                    print(f"Only found {len(rules)} rules, trying lower confidence...")
                    
            except Exception as e:
                print(f"Error generating rules with confidence {threshold}: {e}")
                continue
        
        if rules is None or len(rules) == 0:
            print("Could not generate association rules.")
            return frequent_itemsets, None
        
        # Sort rules by confidence and lift
        rules = rules.sort_values(['confidence', 'lift'], ascending=False)
        
        print(f"Final: Generated {len(rules)} association rules")
        return frequent_itemsets, rules
    
    def create_recommendation_lookup(self, rules, top_n=5):
        """Create a lookup dictionary for fast recommendations"""
        print("Creating recommendation lookup...")
        
        recommendations = {}
        
        for _, rule in rules.iterrows():
            # Get the antecedent (if user watched this)
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            
            # For single movie antecedents (most common case)
            if len(antecedents) == 1:
                movie = antecedents[0]
                
                if movie not in recommendations:
                    recommendations[movie] = []
                
                for consequent in consequents:
                    recommendations[movie].append({
                        'movie': consequent,
                        'confidence': round(rule['confidence'], 3),
                        'support': round(rule['support'], 3),
                        'lift': round(rule['lift'], 3)
                    })
        
        # Sort recommendations by confidence and keep top N
        for movie in recommendations:
            recommendations[movie] = sorted(
                recommendations[movie], 
                key=lambda x: x['confidence'], 
                reverse=True
            )[:top_n]
        
        print(f"Created recommendations for {len(recommendations)} movies")
        return recommendations
    
    def train_model_optimized(self, min_support=0.05, min_rating=3.5, min_movie_frequency=15, 
                            sample_fraction=0.8, max_movies_per_user=50):
        """Complete training pipeline with balanced parameters"""
        print("Starting optimized model training...")
        print("Using balanced parameters for better results")
        
        # Step 1: Download dataset if not exists
        if not os.path.exists('data/ml-100k'):
            if not self.download_dataset():
                return False
        
        # Step 2: Load data efficiently
        self.load_data_efficiently(sample_fraction=sample_fraction)
        
        # Step 3: Prepare transactions with balanced filtering
        transactions = self.prepare_transaction_data_optimized(
            min_rating=min_rating, 
            min_movie_frequency=min_movie_frequency,
            max_movies_per_user=max_movies_per_user
        )
        
        if len(transactions) < 30:
            print("Too few transactions generated. Try lowering min_movie_frequency.")
            return False
        
        # Step 4: Apply FP-Growth algorithm with adaptive parameters
        self.frequent_itemsets, self.association_rules_df = self.apply_fpgrowth_algorithm(
            transactions, min_support=min_support
        )
        
        # Clean up transactions
        del transactions
        gc.collect()
        
        if self.association_rules_df is not None and len(self.association_rules_df) > 0:
            # Step 5: Create recommendation lookup
            self.movie_recommendations = self.create_recommendation_lookup(
                self.association_rules_df
            )
            
            print("Model training completed successfully!")
            print(f"Final memory usage:")
            self.check_memory()
            return True
        else:
            print("Model training failed - no association rules generated")
            print("Try lowering min_support or min_movie_frequency parameters")
            return False
    
    def get_recommendations(self, movie_title, top_n=5):
        """Get recommendations for a given movie"""
        if movie_title in self.movie_recommendations:
            return self.movie_recommendations[movie_title][:top_n]
        else:
            return []
    
    def save_model(self, filepath='model/movie_recommendations.json'):
        """Save the trained model to JSON file"""
        print("Saving model...")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare data for saving
        model_data = {
            'movie_recommendations': self.movie_recommendations,
            'model_info': {
                'total_movies_with_recommendations': len(self.movie_recommendations),
                'frequent_itemsets_count': len(self.frequent_itemsets) if self.frequent_itemsets is not None else 0,
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='model/movie_recommendations.json'):
        """Load a pre-trained model from JSON file"""
        print("Loading model...")
        
        try:
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            
            self.movie_recommendations = model_data['movie_recommendations']
            
            print(f"Model loaded successfully!")
            print(f"Movies with recommendations: {len(self.movie_recommendations)}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_popular_movies(self, top_n=20):
        """Get most popular movies (those with most recommendations)"""
        if not self.movie_recommendations:
            return []
        
        # Sort movies by number of recommendations they can generate
        movie_popularity = [(movie, len(recs)) 
                          for movie, recs in self.movie_recommendations.items()]
        movie_popularity.sort(key=lambda x: x[1], reverse=True)
        
        return [movie for movie, _ in movie_popularity[:top_n]]
    
    def display_sample_recommendations(self, num_samples=5):
        """Display sample recommendations for demonstration"""
        print("\n" + "="*60)
        print("SAMPLE MOVIE RECOMMENDATIONS")
        print("="*60)
        
        popular_movies = self.get_popular_movies(num_samples)
        
        for movie in popular_movies:
            recommendations = self.get_recommendations(movie)
            if recommendations:
                print(f"\nIf you liked: '{movie}'")
                print("You might also like:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"  {i}. {rec['movie']} (confidence: {rec['confidence']:.1%})")
    
    def search_movies(self, query):
        """Search for movies in the recommendation system"""
        if not self.movie_recommendations:
            return []
        
        query_lower = query.lower()
        matching_movies = [movie for movie in self.movie_recommendations.keys() 
                          if query_lower in movie.lower()]
        return matching_movies[:10]  # Return top 10 matches


# Example usage with balanced parameters
if __name__ == "__main__":
    # Initialize the model
    model = MovieRecommendationModel()
    
    print("="*60)
    print("TRAINING MODEL WITH BALANCED PARAMETERS")
    print("="*60)
    print("Using balanced parameters for better rule generation:")
    print("- Lower minimum support (0.05)")
    print("- Lower movie frequency requirement (15)")
    print("- Include ratings >= 3.5")
    print("- Use 80% of data")
    print("- Allow up to 50 movies per user")
    print("- Adaptive confidence thresholds")
    print()
    
    # Train the model with balanced parameters
    success = model.train_model_optimized(
        min_support=0.05,           # Much lower for more itemsets
        min_rating=3.5,             # Include good ratings, not just excellent
        min_movie_frequency=15,     # Lower threshold for more variety
        sample_fraction=0.8,        # Use more data
        max_movies_per_user=50      # Allow more movies per user
    )
    
    if success:
        # Save the model
        model.save_model()
        
        # Display sample recommendations
        model.display_sample_recommendations()
        
        # Interactive testing
        print("\n" + "="*60)
        print("TESTING RECOMMENDATIONS")
        print("="*60)
        
        # Show some available movies
        popular_movies = model.get_popular_movies(10)
        print("Available movies for testing:")
        for i, movie in enumerate(popular_movies, 1):
            print(f"  {i}. {movie}")
        
        # Test with a few movies
        if popular_movies:
            print(f"\nTesting recommendations for: '{popular_movies[0]}'")
            recs = model.get_recommendations(popular_movies[0])
            for rec in recs:
                print(f"  - {rec['movie']} (confidence: {rec['confidence']:.1%})")
    
    else:
        print("\n" + "="*60)
        print("MODEL TRAINING FAILED")
        print("="*60)
        print("Try these even more relaxed parameters:")
        print("- Decrease min_support to 0.03")
        print("- Decrease min_movie_frequency to 10")
        print("- Decrease min_rating to 3.0")
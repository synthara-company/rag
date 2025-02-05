import os
import json
import time
import requests
import psutil
import tmdbsimple as tmdb
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from memory_profiler import profile
from line_profiler import LineProfiler

# Load API key from environment
from dotenv import load_dotenv
load_dotenv()

class SciFiDatasetCollector:
    def __init__(self):
        """
        Initialize the dataset collector with TMDB API
        """
        # Set TMDB API key from environment variable
        tmdb.API_KEY = os.getenv('TMDB_API_KEY', '')
        
        # Directories for data storage
        self.base_dir = os.path.join(os.path.dirname(__file__), 'datasets')
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Performance tracking
        self.performance_log = []
    
    @profile
    def collect_sci_fi_movies(self, pages: int = 10) -> List[Dict[str, Any]]:
        """
        Collect science fiction movies from TMDB
        
        Args:
            pages (int): Number of pages to collect
        
        Returns:
            List of movie dictionaries
        """
        sci_fi_movies = []
        
        # Start performance tracking
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        try:
            for page in range(1, pages + 1):
                # Fetch sci-fi movies
                discover = tmdb.Discover()
                response = discover.movie(
                    with_genres='878',  # Science Fiction genre code
                    page=page,
                    language='en-US',
                    sort_by='popularity.desc'
                )
                
                # Process each movie
                for movie in response['results']:
                    # Fetch detailed movie information
                    movie_details = tmdb.Movies(movie['id'])
                    info = movie_details.info()
                    
                    # Extract relevant information
                    sci_fi_movie = {
                        'id': movie['id'],
                        'title': movie['title'],
                        'overview': movie['overview'],
                        'release_date': movie.get('release_date', 'Unknown'),
                        'popularity': movie['popularity'],
                        'vote_average': movie['vote_average'],
                        'genres': [genre['name'] for genre in info.get('genres', [])],
                        'production_companies': [
                            company['name'] for company in info.get('production_companies', [])
                        ],
                        'keywords': [
                            keyword['name'] for keyword in movie_details.keywords()['keywords']
                        ]
                    }
                    
                    sci_fi_movies.append(sci_fi_movie)
                
                # Simulate rate limiting
                time.sleep(0.5)
        
        except Exception as e:
            print(f"Error collecting movies: {e}")
        
        # End performance tracking
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Log performance metrics
        performance_metrics = {
            'total_movies': len(sci_fi_movies),
            'collection_time': end_time - start_time,
            'memory_usage': end_memory - start_memory
        }
        self.performance_log.append(performance_metrics)
        
        return sci_fi_movies
    
    def collect_cosmos_content(self) -> List[Dict[str, Any]]:
        """
        Collect cosmos and astronomy-related content from NASA API
        
        Returns:
            List of cosmos-related content
        """
        cosmos_content = []
        
        try:
            # NASA APOD (Astronomy Picture of the Day) API
            nasa_api_key = os.getenv('NASA_API_KEY', '')
            base_url = 'https://api.nasa.gov/planetary/apod'
            
            # Collect multiple days of content
            for days_ago in range(30):
                params = {
                    'api_key': nasa_api_key,
                    'date': time.strftime('%Y-%m-%d', time.localtime(time.time() - days_ago * 86400))
                }
                
                response = requests.get(base_url, params=params)
                if response.status_code == 200:
                    content = response.json()
                    cosmos_content.append({
                        'date': content.get('date'),
                        'title': content.get('title'),
                        'explanation': content.get('explanation'),
                        'media_type': content.get('media_type'),
                        'url': content.get('url')
                    })
                
                time.sleep(0.5)  # Rate limiting
        
        except Exception as e:
            print(f"Error collecting cosmos content: {e}")
        
        return cosmos_content
    
    def save_dataset(self, data: List[Dict[str, Any]], filename: str):
        """
        Save dataset to JSON file
        
        Args:
            data (List[Dict]): Dataset to save
            filename (str): Output filename
        """
        filepath = os.path.join(self.base_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Dataset saved to {filepath}")
    
    def analyze_performance(self):
        """
        Analyze and print performance metrics
        """
        if not self.performance_log:
            print("No performance data available")
            return
        
        # Convert performance log to DataFrame
        df = pd.DataFrame(self.performance_log)
        
        print("\n--- Performance Analysis ---")
        print(f"Total Collections: {len(df)}")
        print(f"Average Collection Time: {df['collection_time'].mean():.2f} seconds")
        print(f"Average Memory Usage: {df['memory_usage'].mean():.2f} MB")
        print(f"Total Movies Collected: {df['total_movies'].sum()}")

def main():
    # Initialize collector
    collector = SciFiDatasetCollector()
    
    # Collect sci-fi movies
    sci_fi_movies = collector.collect_sci_fi_movies(pages=5)
    collector.save_dataset(sci_fi_movies, 'sci_fi_movies.json')
    
    # Collect cosmos content
    cosmos_content = collector.collect_cosmos_content()
    collector.save_dataset(cosmos_content, 'cosmos_content.json')
    
    # Analyze performance
    collector.analyze_performance()

if __name__ == "__main__":
    main()

from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import os
import re
from datetime import datetime
import logging
from functools import lru_cache
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend connection

class MovieRecommendationAPI:
    def __init__(self):
        self.recommendations = {}
        self.model_info = {}
        self.movie_search_index = {}  # For faster searching
        self.stats = {
            'requests_count': 0,
            'last_request': None,
            'popular_searches': {}
        }
        self.load_model()
        self._build_search_index()
    
    def load_model(self, filepath='model/movie_recommendations.json'):
        """Load the pre-trained model"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    model_data = json.load(f)
                
                self.recommendations = model_data.get('movie_recommendations', {})
                self.model_info = model_data.get('model_info', {})
                
                logger.info(f"Model loaded successfully!")
                logger.info(f"Movies with recommendations: {len(self.recommendations)}")
                return True
            else:
                logger.error(f"Model file not found: {filepath}")
                logger.error("Please run the training script first!")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _build_search_index(self):
        """Build search index for faster movie searches"""
        logger.info("Building search index...")
        self.movie_search_index = {}
        
        for movie in self.recommendations.keys():
            # Create searchable tokens
            tokens = self._tokenize_title(movie)
            for token in tokens:
                if token not in self.movie_search_index:
                    self.movie_search_index[token] = []
                self.movie_search_index[token].append(movie)
        
        logger.info(f"Search index built with {len(self.movie_search_index)} tokens")
    
    def _tokenize_title(self, title):
        """Tokenize movie title for better searching"""
        # Remove year and special characters, split into words
        cleaned = re.sub(r'\(\d{4}\)', '', title)  # Remove year
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)  # Remove special chars
        tokens = cleaned.lower().split()
        
        # Also add the full title
        tokens.append(title.lower().strip())
        return [token.strip() for token in tokens if len(token.strip()) > 2]
    
    def _track_request(self, movie_title=None):
        """Track API usage statistics"""
        self.stats['requests_count'] += 1
        self.stats['last_request'] = datetime.now().isoformat()
        
        if movie_title:
            movie_key = movie_title.lower()
            if movie_key not in self.stats['popular_searches']:
                self.stats['popular_searches'][movie_key] = 0
            self.stats['popular_searches'][movie_key] += 1
    
    @lru_cache(maxsize=1000)
    def get_recommendations_cached(self, movie_title, top_n=5):
        """Get recommendations with caching for better performance"""
        return self._get_recommendations_internal(movie_title, top_n)
    
    def _get_recommendations_internal(self, movie_title, top_n=5):
        """Internal method to get recommendations"""
        movie_title = movie_title.strip()
        
        # Exact match first
        if movie_title in self.recommendations:
            return self.recommendations[movie_title][:top_n]
        
        # Try case-insensitive exact match
        for movie in self.recommendations:
            if movie.lower() == movie_title.lower():
                return self.recommendations[movie][:top_n]
        
        # Try partial match with scoring
        matches = []
        query_lower = movie_title.lower()
        
        for movie in self.recommendations:
            movie_lower = movie.lower()
            
            # Exact substring match gets higher score
            if query_lower in movie_lower:
                score = len(query_lower) / len(movie_lower)  # Longer match = higher score
                matches.append((score, movie))
            elif movie_lower in query_lower:
                score = len(movie_lower) / len(query_lower)
                matches.append((score, movie))
        
        if matches:
            # Return recommendations from best match
            matches.sort(reverse=True)  # Sort by score descending
            best_match = matches[0][1]
            return self.recommendations[best_match][:top_n]
        
        return []
    
    def get_recommendations(self, movie_title, top_n=5):
        """Public method to get recommendations with tracking"""
        self._track_request(movie_title)
        return self.get_recommendations_cached(movie_title, top_n)
    
    def get_all_movies(self):
        """Get all movies that have recommendations"""
        return sorted(list(self.recommendations.keys()))
    
    def get_popular_movies(self, top_n=20):
        """Get most popular movies"""
        if not self.recommendations:
            return []
        
        # Sort by number of recommendations
        movie_popularity = [(movie, len(recs)) 
                          for movie, recs in self.recommendations.items()]
        movie_popularity.sort(key=lambda x: x[1], reverse=True)
        
        return [movie for movie, _ in movie_popularity[:top_n]]
    
    def search_movies_advanced(self, query, limit=10):
        """Advanced movie search using the search index"""
        query = query.lower().strip()
        if not query:
            return []
        
        # Use search index for faster results
        scored_movies = {}
        query_tokens = query.split()
        
        for token in query_tokens:
            # Find movies containing this token
            for search_token, movies in self.movie_search_index.items():
                if token in search_token:
                    for movie in movies:
                        if movie not in scored_movies:
                            scored_movies[movie] = 0
                        # Score based on token match length
                        scored_movies[movie] += len(token) / len(search_token)
        
        # Sort by score and return top results
        if scored_movies:
            sorted_movies = sorted(scored_movies.items(), key=lambda x: x[1], reverse=True)
            return [movie for movie, score in sorted_movies[:limit]]
        
        # Fallback to simple search
        return self.search_movies_simple(query, limit)
    
    def search_movies_simple(self, query, limit=10):
        """Simple movie search (fallback)"""
        query = query.lower().strip()
        if not query:
            return []
        
        matches = []
        for movie in self.recommendations.keys():
            if query in movie.lower():
                matches.append(movie)
        
        return matches[:limit]
    
    def get_movie_details(self, movie_title):
        """Get detailed information about a movie"""
        movie_title = movie_title.strip()
        
        # Find the movie (case-insensitive)
        actual_movie = None
        for movie in self.recommendations.keys():
            if movie.lower() == movie_title.lower():
                actual_movie = movie
                break
        
        if not actual_movie:
            return None
        
        recommendations = self.recommendations[actual_movie]
        
        # Extract year if present
        year_match = re.search(r'\((\d{4})\)', actual_movie)
        year = year_match.group(1) if year_match else None
        
        return {
            'title': actual_movie,
            'year': year,
            'recommendations_count': len(recommendations),
            'recommendations': recommendations,
            'similar_movies': [rec['movie'] for rec in recommendations[:3]]
        }
    
    def get_random_movies(self, count=5):
        """Get random movies for exploration"""
        import random
        movies = list(self.recommendations.keys())
        return random.sample(movies, min(count, len(movies)))
    
    def get_stats(self):
        """Get API usage statistics"""
        return {
            **self.stats,
            'top_searches': sorted(
                self.stats['popular_searches'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }

# Initialize the API
rec_api = MovieRecommendationAPI()

@app.before_request
def before_request():
    """Log incoming requests"""
    if request.endpoint and request.endpoint != 'health_check':
        logger.info(f"{request.method} {request.path} - {request.remote_addr}")

@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        'message': 'AI-Powered Movie Recommendation API',
        'status': 'running',
        'version': '2.0',
        'model_loaded': len(rec_api.recommendations) > 0,
        'total_movies': len(rec_api.recommendations),
        'last_updated': rec_api.stats['last_request'],
        'endpoints': {
            'GET /': 'API information',
            'GET /movies': 'Get all available movies',
            'GET /movies/popular?limit=N': 'Get popular movies',
            'GET /movies/random?count=N': 'Get random movies',
            'GET /movies/search?q=query&limit=N': 'Search movies',
            'GET /movies/<title>': 'Get movie details',
            'POST /recommendations': 'Get recommendations (JSON body)',
            'GET /recommendations/<title>?limit=N': 'Get recommendations (URL)',
            'GET /model/info': 'Model information',
            'GET /stats': 'API usage statistics',
            'GET /health': 'Health check'
        }
    })

@app.route('/movies', methods=['GET'])
def get_movies():
    """Get all available movies with pagination"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        
        # Limit per_page to prevent large responses
        per_page = min(per_page, 200)
        
        movies = rec_api.get_all_movies()
        total = len(movies)
        
        # Calculate pagination
        start = (page - 1) * per_page
        end = start + per_page
        paginated_movies = movies[start:end]
        
        return jsonify({
            'success': True,
            'total': total,
            'page': page,
            'per_page': per_page,
            'pages': (total + per_page - 1) // per_page,
            'count': len(paginated_movies),
            'movies': paginated_movies
        })
    except Exception as e:
        logger.error(f"Error in get_movies: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/movies/popular', methods=['GET'])
def get_popular_movies():
    """Get popular movies"""
    try:
        limit = request.args.get('limit', 20, type=int)
        limit = min(limit, 100)  # Cap at 100
        
        popular_movies = rec_api.get_popular_movies(limit)
        
        return jsonify({
            'success': True,
            'count': len(popular_movies),
            'movies': popular_movies
        })
    except Exception as e:
        logger.error(f"Error in get_popular_movies: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/movies/random', methods=['GET'])
def get_random_movies():
    """Get random movies for exploration"""
    try:
        count = request.args.get('count', 5, type=int)
        count = min(count, 20)  # Cap at 20
        
        random_movies = rec_api.get_random_movies(count)
        
        return jsonify({
            'success': True,
            'count': len(random_movies),
            'movies': random_movies
        })
    except Exception as e:
        logger.error(f"Error in get_random_movies: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/movies/search', methods=['GET'])
def search_movies():
    """Search movies by title with advanced search"""
    try:
        query = request.args.get('q', '').strip()
        limit = request.args.get('limit', 10, type=int)
        limit = min(limit, 50)  # Cap at 50
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query parameter "q" is required'
            }), 400
        
        if len(query) < 2:
            return jsonify({
                'success': False,
                'error': 'Query must be at least 2 characters long'
            }), 400
        
        start_time = time.time()
        results = rec_api.search_movies_advanced(query, limit)
        search_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'query': query,
            'count': len(results),
            'search_time': round(search_time * 1000, 2),  # in milliseconds
            'movies': results
        })
        
    except Exception as e:
        logger.error(f"Error in search_movies: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/movies/<movie_title>', methods=['GET'])
def get_movie_details(movie_title):
    """Get detailed information about a specific movie"""
    try:
        details = rec_api.get_movie_details(movie_title)
        
        if details is None:
            return jsonify({
                'success': False,
                'error': 'Movie not found'
            }), 404
        
        return jsonify({
            'success': True,
            'movie_details': details
        })
        
    except Exception as e:
        logger.error(f"Error in get_movie_details: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/recommendations', methods=['POST'])
def get_recommendations_post():
    """Get movie recommendations via POST"""
    try:
        data = request.get_json()
        
        if not data or 'movie' not in data:
            return jsonify({
                'success': False,
                'error': 'Movie title is required in request body'
            }), 400
        
        movie_title = data['movie'].strip()
        if not movie_title:
            return jsonify({
                'success': False,
                'error': 'Movie title cannot be empty'
            }), 400
        
        top_n = min(data.get('limit', 5), 20)  # Cap at 20
        
        start_time = time.time()
        recommendations = rec_api.get_recommendations(movie_title, top_n)
        processing_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'movie': movie_title,
            'count': len(recommendations),
            'processing_time': round(processing_time * 1000, 2),
            'recommendations': recommendations
        })
        
    except Exception as e:
        logger.error(f"Error in get_recommendations_post: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/recommendations/<movie_title>', methods=['GET'])
def get_recommendations_get(movie_title):
    """Get recommendations via GET request"""
    try:
        if not movie_title.strip():
            return jsonify({
                'success': False,
                'error': 'Movie title cannot be empty'
            }), 400
        
        limit = min(request.args.get('limit', 5, type=int), 20)  # Cap at 20
        
        start_time = time.time()
        recommendations = rec_api.get_recommendations(movie_title, limit)
        processing_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'movie': movie_title,
            'count': len(recommendations),
            'processing_time': round(processing_time * 1000, 2),
            'recommendations': recommendations
        })
        
    except Exception as e:
        logger.error(f"Error in get_recommendations_get: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/model/info', methods=['GET'])
def get_model_info():
    """Get model information"""
    return jsonify({
        'success': True,
        'model_info': rec_api.model_info,
        'total_movies_with_recommendations': len(rec_api.recommendations),
        'sample_movies': rec_api.get_popular_movies(5),
        'model_file_exists': os.path.exists('model/movie_recommendations.json')
    })

@app.route('/stats', methods=['GET'])
def get_api_stats():
    """Get API usage statistics"""
    try:
        stats = rec_api.get_stats()
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        logger.error(f"Error in get_api_stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': len(rec_api.recommendations) > 0,
        'total_movies': len(rec_api.recommendations),
        'uptime': time.time() - start_time if 'start_time' in globals() else 0
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': list(app.url_map.iter_rules())
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    start_time = time.time()
    
    if len(rec_api.recommendations) == 0:
        print("\n" + "="*60)
        print("⚠️  WARNING: No model loaded!")
        print("Please run the training script first to create the model.")
        print("Command: python train_model.py")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("🎬 AI-POWERED MOVIE RECOMMENDATION API")
        print("="*60)
        print(f"✅ Model loaded successfully!")
        print(f"📊 Movies with recommendations: {len(rec_api.recommendations)}")
        print(f"🔍 Search index built with {len(rec_api.movie_search_index)} tokens")
        print("\n🌐 API Endpoints:")
        print("  GET  /                         - API info")
        print("  GET  /movies                   - All movies (paginated)")
        print("  GET  /movies/popular           - Popular movies")
        print("  GET  /movies/random            - Random movies")
        print("  GET  /movies/search?q=...      - Search movies")
        print("  GET  /movies/<title>           - Movie details")
        print("  POST /recommendations          - Get recommendations (JSON)")
        print("  GET  /recommendations/<movie>  - Get recommendations (URL)")
        print("  GET  /model/info               - Model information")
        print("  GET  /stats                    - API usage statistics")
        print("  GET  /health                   - Health check")
        print("\n🚀 Starting server...")
        print("="*60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
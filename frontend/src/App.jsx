import React, { useState, useEffect } from "react";
import { Film, Star, Shuffle, ArrowLeft, Search, Clock } from "lucide-react";

const API_BASE = "http://localhost:5000";

const MovieCard = ({ movie, onClick, showYear = true }) => {
  const getYear = (title) => {
    const match = title.match(/\((\d{4})\)/);
    return match ? match[1] : "";
  };

  const getTitle = (title) => {
    return title.replace(/\s*\(\d{4}\)\s*$/, "");
  };

  return (
    <div
      onClick={() => onClick(movie)}
      className="bg-gray-800 rounded-lg p-4 hover:bg-gray-700 transition-all cursor-pointer hover:scale-105 border border-gray-700"
    >
      <div className="flex items-center gap-3">
        <div className="bg-gray-700 p-2 rounded-lg">
          <Film className="w-6 h-6 text-blue-400" />
        </div>
        <div className="flex-1">
          <h3 className="text-white font-medium text-sm leading-tight">
            {getTitle(movie)}
          </h3>
          {showYear && getYear(movie) && (
            <p className="text-gray-400 text-xs mt-1">{getYear(movie)}</p>
          )}
        </div>
      </div>
    </div>
  );
};

const MovieSection = ({
  title,
  movies,
  onMovieClick,
  icon: Icon,
  isLoading,
}) => (
  <div className="mb-8">
    <div className="flex items-center gap-2 mb-4">
      <Icon className="w-5 h-5 text-blue-400" />
      <h2 className="text-xl font-semibold text-white">{title}</h2>
    </div>

    {isLoading ? (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {[...Array(6)].map((_, i) => (
          <div key={i} className="bg-gray-800 rounded-lg p-4 animate-pulse">
            <div className="flex items-center gap-3">
              <div className="bg-gray-700 p-2 rounded-lg">
                <div className="w-6 h-6 bg-gray-600 rounded"></div>
              </div>
              <div className="flex-1">
                <div className="h-4 bg-gray-700 rounded mb-2"></div>
                <div className="h-3 bg-gray-700 rounded w-1/3"></div>
              </div>
            </div>
          </div>
        ))}
      </div>
    ) : (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {movies.map((movie, index) => (
          <MovieCard key={index} movie={movie} onClick={onMovieClick} />
        ))}
      </div>
    )}
  </div>
);

const SearchBar = ({ onSearch, searchQuery, setSearchQuery }) => {
  const [isSearching, setIsSearching] = useState(false);

  const handleSearch = async (query) => {
    if (!query.trim()) return;

    setIsSearching(true);
    try {
      await onSearch(query);
    } finally {
      setIsSearching(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      handleSearch(searchQuery);
    }
  };

  return (
    <div className="mb-8">
      <div className="flex gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyUp={handleKeyPress}
            placeholder="Search for movies..."
            className="w-full bg-gray-800 text-white pl-10 pr-4 py-3 rounded-lg border border-gray-700 focus:border-blue-400 focus:outline-none"
            disabled={isSearching}
          />
        </div>
        <button
          onClick={() => handleSearch(searchQuery)}
          disabled={isSearching || !searchQuery.trim()}
          className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed px-6 py-3 rounded-lg text-white font-medium transition-colors"
        >
          {isSearching ? "Searching..." : "Search"}
        </button>
      </div>
    </div>
  );
};

const HomePage = ({ onMovieSelect }) => {
  const [popularMovies, setPopularMovies] = useState([]);
  const [randomMovies, setRandomMovies] = useState([]);
  const [searchResults, setSearchResults] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [loading, setLoading] = useState({
    popular: true,
    random: true,
    search: false,
  });
  const [showSearch, setShowSearch] = useState(false);

  useEffect(() => {
    loadPopularMovies();
    loadRandomMovies();
  }, []);

  const loadPopularMovies = async () => {
    try {
      const response = await fetch(`${API_BASE}/movies/popular?limit=6`);
      const data = await response.json();
      setPopularMovies(data.movies || []);
    } catch (error) {
      console.error("Error loading popular movies:", error);
    } finally {
      setLoading((prev) => ({ ...prev, popular: false }));
    }
  };

  const loadRandomMovies = async () => {
    try {
      const response = await fetch(`${API_BASE}/movies/random?count=6`);
      const data = await response.json();
      setRandomMovies(data.movies || []);
    } catch (error) {
      console.error("Error loading random movies:", error);
    } finally {
      setLoading((prev) => ({ ...prev, random: false }));
    }
  };

  const handleSearch = async (query) => {
    setLoading((prev) => ({ ...prev, search: true }));
    try {
      const response = await fetch(
        `${API_BASE}/movies/search?q=${encodeURIComponent(query)}`
      );
      const data = await response.json();
      setSearchResults(data.movies || []);
      setShowSearch(true);
    } catch (error) {
      console.error("Error searching movies:", error);
      setSearchResults([]);
    } finally {
      setLoading((prev) => ({ ...prev, search: false }));
    }
  };

  const clearSearch = () => {
    setSearchQuery("");
    setSearchResults([]);
    setShowSearch(false);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-2 mb-2">
            <Film className="w-8 h-8 text-blue-400" />
            <h1 className="text-3xl font-bold text-white">MovieMind</h1>
          </div>
          <p className="text-gray-400">Discover your next favorite movie</p>
        </div>

        {/* Search */}
        <SearchBar
          onSearch={handleSearch}
          searchQuery={searchQuery}
          setSearchQuery={setSearchQuery}
        />

        {/* Search Results */}
        {showSearch && (
          <div className="mb-8">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Search className="w-5 h-5 text-blue-400" />
                <h2 className="text-xl font-semibold text-white">
                  Search Results for "{searchQuery}"
                </h2>
              </div>
              <button
                onClick={clearSearch}
                className="text-gray-400 hover:text-white transition-colors"
              >
                Clear
              </button>
            </div>

            {loading.search ? (
              <div className="text-center py-8">
                <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400"></div>
                <p className="text-gray-400 mt-2">Searching...</p>
              </div>
            ) : searchResults.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {searchResults.map((movie, index) => (
                  <MovieCard
                    key={index}
                    movie={movie}
                    onClick={onMovieSelect}
                  />
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-400">
                No movies found for "{searchQuery}"
              </div>
            )}
          </div>
        )}

        {/* Popular Movies */}
        <MovieSection
          title="Popular Movies"
          movies={popularMovies}
          onMovieClick={onMovieSelect}
          icon={Star}
          isLoading={loading.popular}
        />

        {/* Random Movies */}
        <MovieSection
          title="Discover Something New"
          movies={randomMovies}
          onMovieClick={onMovieSelect}
          icon={Shuffle}
          isLoading={loading.random}
        />

        {/* Refresh Random Button */}
        <div className="text-center mt-6">
          <button
            onClick={loadRandomMovies}
            disabled={loading.random}
            className="bg-gray-800 hover:bg-gray-700 disabled:bg-gray-600 px-6 py-2 rounded-lg text-white font-medium transition-colors border border-gray-700"
          >
            {loading.random ? "Loading..." : "Get More Random Movies"}
          </button>
        </div>
      </div>
    </div>
  );
};

const MovieDetailPage = ({ movie, onBack }) => {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadRecommendations();
  }, [movie]);

  const loadRecommendations = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE}/recommendations/${encodeURIComponent(movie)}`
      );
      const data = await response.json();

      if (response.ok) {
        setRecommendations(data.recommendations || []);
      } else {
        setError(data.error || "Failed to load recommendations");
      }
    } catch (error) {
      console.error("Error loading recommendations:", error);
      setError("Failed to connect to server");
    } finally {
      setLoading(false);
    }
  };

  const getYear = (title) => {
    const match = title.match(/\((\d{4})\)/);
    return match ? match[1] : "";
  };

  const getTitle = (title) => {
    return title.replace(/\s*\(\d{4}\)\s*$/, "");
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="container mx-auto px-4 py-8">
        {/* Back Button */}
        <button
          onClick={onBack}
          className="flex items-center gap-2 text-blue-400 hover:text-blue-300 mb-6 transition-colors"
        >
          <ArrowLeft className="w-5 h-5" />
          Back to Home
        </button>

        {/* Movie Details */}
        <div className="bg-gray-800 rounded-lg p-8 mb-8">
          <div className="flex items-start gap-6">
            <div className="bg-gray-700 p-4 rounded-lg flex-shrink-0">
              <Film className="w-16 h-16 text-blue-400" />
            </div>
            <div className="flex-1">
              <h1 className="text-3xl font-bold text-white mb-2">
                {getTitle(movie)}
              </h1>
              {getYear(movie) && (
                <div className="flex items-center gap-2 text-gray-400 mb-4">
                  <Clock className="w-4 h-4" />
                  <span>{getYear(movie)}</span>
                </div>
              )}
              <p className="text-gray-300">
                You've selected this movie. Check out our recommendations below
                based on what other users who liked this movie also enjoyed!
              </p>
            </div>
          </div>
        </div>

        {/* Recommendations */}
        <div>
          <div className="flex items-center gap-2 mb-6">
            <Star className="w-6 h-6 text-blue-400" />
            <h2 className="text-2xl font-semibold text-white">
              Recommended for You
            </h2>
          </div>

          {loading ? (
            <div className="text-center py-12">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400 mb-4"></div>
              <p className="text-gray-400">
                Finding perfect recommendations...
              </p>
            </div>
          ) : error ? (
            <div className="bg-gray-800 rounded-lg p-8 text-center">
              <div className="text-red-400 mb-2">⚠️ {error}</div>
              <p className="text-gray-400 mb-4">
                We couldn't find recommendations for this movie.
              </p>
              <button
                onClick={loadRecommendations}
                className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg text-white transition-colors"
              >
                Try Again
              </button>
            </div>
          ) : recommendations.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {recommendations.map((rec, index) => (
                <div
                  key={index}
                  className="bg-gray-800 rounded-lg p-4 border border-gray-700 hover:border-blue-400 transition-colors"
                >
                  <div className="flex items-center gap-3 mb-3">
                    <div className="bg-gray-700 p-2 rounded-lg">
                      <Film className="w-6 h-6 text-blue-400" />
                    </div>
                    <div className="flex-1">
                      <h3 className="text-white font-medium text-sm leading-tight">
                        {getTitle(rec.movie)}
                      </h3>
                      {getYear(rec.movie) && (
                        <p className="text-gray-400 text-xs mt-1">
                          {getYear(rec.movie)}
                        </p>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-gray-400">Confidence</span>
                    <span className="text-blue-400 font-medium">
                      {(rec.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="bg-gray-800 rounded-lg p-8 text-center">
              <div className="text-gray-400 mb-2">🎬</div>
              <p className="text-gray-400">
                No recommendations available for this movie yet.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const App = () => {
  const [currentView, setCurrentView] = useState("home");
  const [selectedMovie, setSelectedMovie] = useState(null);

  const handleMovieSelect = (movie) => {
    setSelectedMovie(movie);
    setCurrentView("detail");
  };

  const handleBackToHome = () => {
    setCurrentView("home");
    setSelectedMovie(null);
  };

  return (
    <div>
      {currentView === "home" ? (
        <HomePage onMovieSelect={handleMovieSelect} />
      ) : (
        <MovieDetailPage movie={selectedMovie} onBack={handleBackToHome} />
      )}
    </div>
  );
};

export default App;

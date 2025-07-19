const Movie = require("../models/Movie");
const WatchHistory = require("../models/WatchHistory");

// Add a new movie to the catalog
const addMovie = async (req, res) => {
  try {
    const { title, genre } = req.body;
    const movie = new Movie({ title, genre });
    await movie.save();
    res.json({ message: "Movie added ✅", movie });
  } catch (err) {
    res.status(500).json({ error: "Failed to add movie" });
  }
};

// Get all movies
const getMovies = async (req, res) => {
  try {
    const movies = await Movie.find();
    res.json(movies);
  } catch (err) {
    res.status(500).json({ error: "Failed to fetch movies" });
  }
};

// Add movie to user's watch history
const addToWatchHistory = async (req, res) => {
  try {
    const { userId, title } = req.body;

    let history = await WatchHistory.findOne({ userId });

    if (!history) {
      history = new WatchHistory({ userId, watched: [title] });
    } else {
      // avoid duplicates
      if (!history.watched.includes(title)) {
        history.watched.push(title);
      }
    }

    await history.save();
    res.json({ message: "Movie added to watch history ✅", history });
  } catch (err) {
    res.status(500).json({ error: "Failed to update watch history" });
  }
};

// Get watch history for a user
const getWatchHistory = async (req, res) => {
  try {
    const { id } = req.params;
    const history = await WatchHistory.findOne({ userId: id });

    if (!history) {
      return res.json({ userId: id, watched: [] });
    }

    res.json(history);
  } catch (err) {
    res.status(500).json({ error: "Failed to fetch watch history" });
  }
};

module.exports = {
  addMovie,
  getMovies,
  addToWatchHistory,
  getWatchHistory,
};

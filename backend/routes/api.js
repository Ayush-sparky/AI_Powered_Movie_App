const express = require("express");
const router = express.Router();
const {
  addMovie,
  getMovies,
  addToWatchHistory,
  getWatchHistory,
} = require("../controllers/movieController");

// Movie catalog
router.post("/movies", addMovie);
router.get("/movies", getMovies);

// Watch history
router.post("/watch", addToWatchHistory);
router.get("/watch/:id", getWatchHistory);

module.exports = router;

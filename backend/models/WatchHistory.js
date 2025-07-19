const mongoose = require("mongoose");

const WatchHistorySchema = new mongoose.Schema({
  userId: {
    type: String, // Keep it string for simplicity (e.g., "user1", "user2")
    required: true,
  },
  watched: {
    type: [String], // Array of movie titles
    default: [],
  },
});

module.exports = mongoose.model("WatchHistory", WatchHistorySchema);

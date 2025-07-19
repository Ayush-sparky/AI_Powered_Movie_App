const mongoose = require("mongoose");

const MovieSchema = new mongoose.Schema({
  title: {
    type: String,
    required: true,
  },
  genre: String,
});

module.exports = mongoose.model("Movie", MovieSchema);

const express = require("express");
const cors = require("cors");
const dotenv = require("dotenv");
const connectDB = require("./config/db");

dotenv.config();
const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// DB Connection
connectDB();

// Routes
app.use("/api", require("./routes/api"));

app.get("/", (req, res) => {
  res.send("Movie Recommendation Backend is Running 🎬");
});

app.listen(PORT, () => {
  console.log(`Server started on port ${PORT}`);
});

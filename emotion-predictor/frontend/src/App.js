import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [colors, setColors] = useState([]);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      setError("Please select a file");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(
        "http://localhost:8000/predict",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      setPrediction(response.data.predicted_emotion);
      setColors(response.data.colors);
      setError(null);
    } catch (error) {
      console.error("Error:", error);
      setError("An error occurred while predicting. Please try again.");
    }
  };

  return (
    <div className="App">
      <h1>Image Emotion Predictor</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleFileChange} />
        <button type="submit">Predict Emotion</button>
      </form>
      {error && <p className="error">{error}</p>}
      {prediction && (
        <div className="results">
          <h2>Predicted Emotion: {prediction}</h2>
          <h3>Primary Colors:</h3>
          <div className="colors">
            {colors.map((color, index) => (
              <div key={index}>
                <div
                  className="color-box"
                  style={{
                    backgroundColor: `rgb(${color[0]}, ${color[1]}, ${color[2]})`,
                  }}
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

import React, { useState } from "react";
import axios from "axios";

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

    // Reset state before new prediction
    setPrediction(null);
    setColors([]);
    setError(null);

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
      console.log("Response data:", response.data);
      setPrediction(response.data.predicted_emotion);
      setColors(response.data.colors);
      setError(null);
    } catch (error) {
      console.error("Error:", error);
      setError("An error occurred while predicting. Please try again.");
      setPrediction(null);
      setColors([]);
    }
  };

  return (
    <div className="App">
      <h1>Image Emotion Predictor</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleFileChange} />
        <button type="submit">Predict Emotion</button>
      </form>
      {error && <p style={{ color: "red" }}>{error}</p>}
      {prediction && (
        <div>
          <h2>Predicted Emotion: {prediction}</h2>
          <h3>Primary Colors:</h3>
          <div style={{ display: "flex" }}>
            {colors && colors.length > 0 ? (
              colors.map((color, index) => (
                <div
                  key={index}
                  style={{
                    width: "50px",
                    height: "50px",
                    backgroundColor: `rgb(${color[0]}, ${color[1]}, ${color[2]})`,
                    marginRight: "10px",
                  }}
                />
              ))
            ) : (
              <p>No colors to display.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

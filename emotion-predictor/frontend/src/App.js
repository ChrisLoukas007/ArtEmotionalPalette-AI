import React, { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [colors, setColors] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) return;

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
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div className="App">
      <h1>Image Emotion Predictor</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleFileChange} />
        <button type="submit">Predict Emotion</button>
      </form>
      {prediction && (
        <div>
          <h2>Predicted Emotion: {prediction}</h2>
          <h3>Primary Colors:</h3>
          <div style={{ display: "flex" }}>
            {colors.map((color, index) => (
              <div
                key={index}
                style={{
                  width: "50px",
                  height: "50px",
                  backgroundColor: `rgb(${color[0]}, ${color[1]}, ${color[2]})`,
                }}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [colors, setColors] = useState([]);
  const [error, setError] = useState(null);
  const [filePreview, setFilePreview] = useState(null);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);

    if (selectedFile) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setFilePreview(reader.result);
      };
      reader.readAsDataURL(selectedFile);
    } else {
      setFilePreview(null);
    }
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
      setPredictions(response.data.predicted_emotions);
      setColors(response.data.colors);
      setError(null);
    } catch (error) {
      console.error("Error:", error);
      setError("An error occurred while predicting. Please try again.");
      setPredictions([]);
    }
  };

  return (
    <div className="App">
      <h1>Image Emotion Predictor</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleFileChange} />
        <button type="submit">Predict Emotion</button>
      </form>
      {filePreview && (
        <div className="image-preview">
          <p>Image Preview:</p>
          <img src={filePreview} alt="Preview" className="preview-image" />
        </div>
      )}
      {error && <p className="error">{error}</p>}
      {predictions.length > 0 && (
        <div className="results">
          <h2>Predicted Emotions:</h2>
          <ul>
            {predictions.map((item, index) => (
              <li key={index}>
                {item.emotion} ({(item.probability * 100).toFixed(2)}%)
              </li>
            ))}
          </ul>
          <h3>Primary Colors:</h3>
          <div className="colors">
            {colors.map((color, index) => (
              <div key={index}>
                <div
                  className="color-box"
                  style={{
                    backgroundColor: `rgb(${color.rgb[0]}, ${color.rgb[1]}, ${color.rgb[2]})`,
                  }}
                />
                <p className="color-info">
                  RGB: ({color.rgb[0]}, {color.rgb[1]}, {color.rgb[2]})<br />
                  Name: {color.name}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

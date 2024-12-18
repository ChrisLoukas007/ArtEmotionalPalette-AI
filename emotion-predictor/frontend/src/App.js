import React, { useState } from "react";
import axios from "axios";
import "./App.css";
import { Spinner } from "react-bootstrap";

function App() {
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [colors, setColors] = useState([]);
  const [error, setError] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modelType, setModelType] = useState("mlp"); // New state for model type

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

  const handleModelChange = (event) => {
    setModelType(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      setError("Please select a file");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(
        `http://localhost:8000/predict?model_type=${modelType}`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      setPredictions(response.data.predicted_emotions);
      setColors(response.data.colors);
    } catch (error) {
      if (error.response && error.response.data && error.response.data.detail) {
        setError(`Error: ${error.response.data.detail}`);
      } else {
        setError("An error occurred while predicting. Please try again.");
      }
      setPredictions([]);
      setColors([]);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPredictions([]);
    setColors([]);
    setError(null);
    setFilePreview(null);
  };

  return (
    <div className="App container mt-5">
      <h1 className="text-center mb-4">Image Emotion Predictor</h1>

      <form onSubmit={handleSubmit} className="text-center">
        <div className="mb-3">
          <input
            type="file"
            className="form-control"
            onChange={handleFileChange}
          />
        </div>
        <div className="mb-3">
          <select
            className="form-select"
            value={modelType}
            onChange={handleModelChange}
          >
            <option value="mlp">MLP Model</option>
            <option value="svm">SVM Model</option>
            <option value="random_forest">Random Forest Model</option>
          </select>
        </div>
        <div>
          <button type="submit" className="btn btn-primary me-2">
            Predict Emotion
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={handleReset}
          >
            Reset
          </button>
        </div>
      </form>

      {filePreview && (
        <div className="image-preview mt-4 text-center">
          <h5>Image Preview:</h5>
          <img
            src={filePreview}
            alt="Preview"
            className="img-fluid preview-image"
            style={{ maxWidth: "300px", height: "auto" }}
          />
        </div>
      )}

      {loading && (
        <div className="text-center mt-3">
          <Spinner animation="border" role="status">
            <span className="visually-hidden">Loading...</span>
          </Spinner>
        </div>
      )}

      {error && <p className="text-danger mt-3 text-center">{error}</p>}

      {predictions.length > 0 && (
        <div className="results mt-4 text-center">
          <h2>Predicted Emotions</h2>
          <ul className="list-group mb-3 text-center">
            {predictions.map((item, index) => (
              <li key={index} className="list-group-item">
                {item.emotion} ({(item.probability * 100).toFixed(2)}%)
              </li>
            ))}
          </ul>
          <h3>Primary Colors</h3>
          <div className="colors d-flex justify-content-center flex-wrap mt-3">
            {colors.map((color, index) => (
              <div key={index} className="color-card text-center mx-2 my-3">
                <div
                  className="color-box"
                  style={{
                    backgroundColor: `rgb(${color.rgb[0]}, ${color.rgb[1]}, ${color.rgb[2]})`,
                  }}
                ></div>
                <div className="color-info mt-2">
                  <strong>RGB:</strong> ({color.rgb[0]}, {color.rgb[1]},{" "}
                  {color.rgb[2]})<br />
                  <strong>{color.name}</strong>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

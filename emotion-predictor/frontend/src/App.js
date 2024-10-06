import React, { useState } from "react";
import axios from "axios";
import "./App.css";
import { Spinner } from "react-bootstrap"; // Import Bootstrap Spinner

function App() {
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [colors, setColors] = useState([]);
  const [error, setError] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  const [loading, setLoading] = useState(false); // New loading state

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
    setLoading(true); // Show spinner when the request starts
    setError(null);

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
      if (error.response && error.response.data && error.response.data.detail) {
        setError(`Error: ${error.response.data.detail}`);
      } else {
        setError("An error occurred while predicting. Please try again.");
      }
      setPredictions([]);
    } finally {
      setLoading(false); // Hide spinner when the request finishes
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
          <h2 className="text-center">Predicted Emotions</h2>
          <ul className="list-group mb-3 text-center">
            {predictions.map((item, index) => (
              <li key={index} className="list-group-item">
                {item.emotion} ({(item.probability * 100).toFixed(2)}%)
              </li>
            ))}
          </ul>
          <h3 className="text-center">Primary Colors</h3>
          <div className="colors d-flex justify-content-center flex-wrap mt-3">
            {colors.map((color, index) => (
              <div key={index} className="color-box mx-2">
                <div
                  className="color-box"
                  style={{
                    width: "50px",
                    height: "50px",
                    backgroundColor: `rgb(${color.rgb[0]}, ${color.rgb[1]}, ${color.rgb[2]})`,
                    border: "1px solid #ccc",
                  }}
                ></div>
                <p className="text-center mt-2">
                  <strong>RGB:</strong> ({color.rgb[0]}, {color.rgb[1]},{" "}
                  {color.rgb[2]})
                  <br />
                  <strong>{color.name}</strong>
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

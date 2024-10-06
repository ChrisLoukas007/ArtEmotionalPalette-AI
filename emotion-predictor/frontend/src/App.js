import React, { useState } from "react";
import axios from "axios";
import "./App.css";
import { Spinner, Card, Button } from "react-bootstrap";

function App() {
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [colors, setColors] = useState([]);
  const [error, setError] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  const [loading, setLoading] = useState(false);

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
    setLoading(true);
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

      <form onSubmit={handleSubmit} className="text-center mb-4">
        <div className="mb-3">
          <input
            type="file"
            className="form-control"
            onChange={handleFileChange}
          />
        </div>
        <div>
          <Button type="submit" variant="primary" className="me-2">
            Predict Emotion
          </Button>
          <Button variant="secondary" onClick={handleReset}>
            Reset
          </Button>
        </div>
      </form>

      {filePreview && (
        <div className="image-preview mt-4 text-center">
          <h5>Image Preview:</h5>
          <img
            src={filePreview}
            alt="Preview"
            className="img-fluid preview-image rounded"
            style={{
              maxWidth: "400px",
              height: "auto",
              border: "1px solid #ddd",
              padding: "10px",
            }}
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
        <div className="results mt-5">
          <h2 className="text-center">Predicted Emotions</h2>
          <ul className="list-group mb-4">
            {predictions.map((item, index) => (
              <li
                key={index}
                className="list-group-item text-center"
                style={{ padding: "10px" }}
              >
                <strong>{item.emotion}</strong> (
                {(item.probability * 100).toFixed(2)}%)
              </li>
            ))}
          </ul>

          <h3 className="text-center">Primary Colors</h3>
          <div className="colors d-flex justify-content-center flex-wrap mt-3">
            {colors.map((color, index) => (
              <Card key={index} className="m-2" style={{ width: "10rem" }}>
                <div
                  className="color-box"
                  style={{
                    width: "100%",
                    height: "100px",
                    backgroundColor: `rgb(${color.rgb[0]}, ${color.rgb[1]}, ${color.rgb[2]})`,
                  }}
                ></div>
                <Card.Body className="text-center">
                  <Card.Text>
                    <strong>Color Name</strong> {color.name} <br />{" "}
                    <strong>RGB</strong> ({color.rgb[0]}, {color.rgb[1]},
                    {color.rgb[2]})
                  </Card.Text>
                </Card.Body>
              </Card>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
